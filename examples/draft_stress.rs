//! Long-document drafting stress test against DeepSeek (OpenAI-compatible
//! API). Mirrors `gemini-potion/apps/editor-demo/scripts/draft-stress.ts` —
//! same 15 agreement prompts (read from the shared JSON), same CSV schema.
//!
//! Run with:
//!
//!   DEEPSEEK_API_KEY=sk-… cargo run --release --example draft_stress
//!
//! Optional env:
//!
//!   DEEPSEEK_BASE_URL    default https://api.deepseek.com/v1/
//!   DEEPSEEK_MODEL       default deepseek-v4-pro
//!   AGREEMENTS_JSON      path to the shared agreements.json. Default looks
//!                        in ../gemini-potion (sibling repo).
//!   DRAFT_CSV            CSV output path. Default ./runs/drafting.csv
//!   DRAFT_BACKEND        backend label written to the CSV. Default "deepseek".

use openai_api_rust::{
    chat::{ChatApi, ChatBody},
    Auth, Message, OpenAI, Role,
};
use serde::Deserialize;
use std::env;
use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Deserialize)]
struct Agreement {
    slug: String,
    title: String,
    context: String,
    sections: Vec<String>,
}

fn iso_now() -> String {
    // ISO 8601 with milliseconds, UTC. Same shape as JS `Date#toISOString()`
    // so the rows line up with the TS-written CSV.
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let total_ms = dur.as_millis() as i64;
    let seconds = total_ms / 1000;
    let ms = (total_ms % 1000) as i32;

    // Plain epoch → UTC date math (no chrono dep).
    let days_since_epoch = seconds / 86_400;
    let secs_of_day = seconds % 86_400;
    let h = secs_of_day / 3600;
    let m = (secs_of_day % 3600) / 60;
    let s = secs_of_day % 60;

    let (y, mo, d) = days_to_ymd(days_since_epoch);
    format!(
        "{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}.{ms:03}Z"
    )
}

fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    // Howard Hinnant's date algorithm — converts days since 1970-01-01 to
    // (year, month, day). Handles leap years correctly without a chrono dep.
    let z = days + 719_468;
    let era = if z >= 0 { z / 146_097 } else { (z - 146_096) / 146_097 };
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if m <= 2 { y + 1 } else { y } as i32;
    (y, m, d)
}

const NANOID_ALPHABET: &[u8] =
    b"useandom-26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict";

fn nanoid() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut out = String::with_capacity(21);
    let mut h = DefaultHasher::new();
    // Mix the system clock + a counter so successive calls in the same ns
    // still differ. The Hasher is just a fast PRNG-ish source for this demo.
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    h.write_u128(now);
    for i in 0..21u8 {
        h.write_u8(i);
        let v = h.finish();
        out.push(NANOID_ALPHABET[(v as usize) & 63] as char);
    }
    out
}

fn csv_cell(v: &str) -> String {
    if v.contains(',') || v.contains('"') || v.contains('\n') {
        format!("\"{}\"", v.replace('"', "\"\""))
    } else {
        v.to_string()
    }
}

const CSV_HEADER: &str =
    "nanoid,date,timestamp,timestamp_completion,backend,model,agreement,ok,ttft_ms,total_ms,chars,sections,error\n";

fn ensure_csv(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }
    if !path.exists() {
        std::fs::write(path, CSV_HEADER)?;
    }
    Ok(())
}

fn append_row(
    path: &Path,
    id: &str,
    started: &str,
    completed: &str,
    backend: &str,
    model: &str,
    agreement: &str,
    ok: bool,
    ttft_ms: u128,
    total_ms: u128,
    chars: usize,
    sections: usize,
    error: &str,
) -> std::io::Result<()> {
    let date = &started[..10]; // YYYY-MM-DD prefix
    let row = [
        csv_cell(id),
        csv_cell(date),
        csv_cell(started),
        csv_cell(completed),
        csv_cell(backend),
        csv_cell(model),
        csv_cell(agreement),
        csv_cell(if ok { "true" } else { "false" }),
        ttft_ms.to_string(),
        total_ms.to_string(),
        chars.to_string(),
        sections.to_string(),
        csv_cell(error),
    ]
    .join(",");
    let mut f = OpenOptions::new().append(true).open(path)?;
    writeln!(f, "{row}")?;
    Ok(())
}

fn build_prompt(a: &Agreement) -> String {
    let section_list = a
        .sections
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{}. {s}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "Draft the full text of a \"{title}\". This is a serious legal document — write the full body, \
not a summary or outline. Use proper legal prose, defined terms in initial caps, and numbered sections. \
You MUST produce at least {n} numbered sections, in this order, each with substantive body text:\n\n\
{section_list}\n\n\
Context for the parties and deal:\n{context}\n\n\
Output formatting:\n\
- Start with a one-paragraph preamble naming the parties and effective date.\n\
- Each section starts with a Markdown heading \"## N. Section Title\".\n\
- No table of contents, no commentary, no code fences — produce only the agreement body.",
        title = a.title,
        n = a.sections.len(),
        section_list = section_list,
        context = a.context,
    )
}

fn count_sections(text: &str) -> usize {
    text.lines().filter(|l| l.starts_with("## ")).count()
}

fn percentile(xs: &mut [u128], p: f64) -> u128 {
    if xs.is_empty() {
        return 0;
    }
    xs.sort_unstable();
    let idx = ((xs.len() as f64 - 1.0) * p).round() as usize;
    xs[idx]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key =
        env::var("DEEPSEEK_API_KEY").map_err(|_| "DEEPSEEK_API_KEY must be set")?;
    let base_url = env::var("DEEPSEEK_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com/v1/".to_string());
    let model = env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-v4-pro".to_string());
    let backend = env::var("DRAFT_BACKEND").unwrap_or_else(|_| "deepseek".to_string());
    let csv_path = env::var("DRAFT_CSV").unwrap_or_else(|_| "runs/drafting.csv".to_string());
    let csv_path = Path::new(&csv_path).to_path_buf();
    let agreements_path = env::var("AGREEMENTS_JSON").unwrap_or_else(|_| {
        // Default: look at the sibling gemini-potion repo.
        "../gemini-potion/apps/editor-demo/fixtures/drafting-agreements.json".to_string()
    });

    let agreements: Vec<Agreement> = serde_json::from_str(&std::fs::read_to_string(&agreements_path)?)?;
    if agreements.is_empty() {
        return Err("agreements.json was empty".into());
    }

    ensure_csv(&csv_path)?;

    println!("# drafting against {base_url} (backend={backend}, model={model})");
    println!("# agreements: {} (from {agreements_path})", agreements.len());
    println!("# csv:        {}", csv_path.display());
    println!();

    let client = OpenAI::new(Auth::new(&api_key), &base_url);
    let overall = Instant::now();
    let mut results: Vec<(String, u128, usize, usize, bool)> = Vec::new();

    for a in &agreements {
        print!("  drafting {:<20} ... ", a.slug);
        std::io::stdout().flush().ok();

        let prompt = build_prompt(a);
        let body = ChatBody {
            model: model.clone(),
            // Long-document budget — single big response with no streaming.
            max_tokens: Some(16384),
            temperature: Some(0.2),
            top_p: None,
            n: Some(1),
            stream: Some(false),
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            messages: vec![Message::new(Role::User, prompt)],
            tools: None,
            tool_choice: None,
        };

        let started_iso = iso_now();
        let t0 = Instant::now();
        let result = client.chat_completion_create(&body);
        let elapsed = t0.elapsed().as_millis();
        let completed_iso = iso_now();

        match result {
            Ok(completion) => {
                let text = completion
                    .choices
                    .into_iter()
                    .next()
                    .and_then(|c| c.message)
                    .map(|m| m.content)
                    .unwrap_or_default();
                let chars = text.chars().count();
                let sections = count_sections(&text);
                println!(
                    "ttft={el}ms total={el}ms chars={chars} sections={sections}/{want}",
                    el = elapsed,
                    want = a.sections.len()
                );
                results.push((a.slug.clone(), elapsed, chars, sections, true));
                append_row(
                    &csv_path,
                    &nanoid(),
                    &started_iso,
                    &completed_iso,
                    &backend,
                    &model,
                    &a.slug,
                    true,
                    elapsed,
                    elapsed,
                    chars,
                    sections,
                    "",
                )?;
            }
            Err(e) => {
                let msg = format!("{e:?}");
                println!("FAILED after {elapsed}ms: {msg}");
                results.push((a.slug.clone(), elapsed, 0, 0, false));
                append_row(
                    &csv_path,
                    &nanoid(),
                    &started_iso,
                    &completed_iso,
                    &backend,
                    &model,
                    &a.slug,
                    false,
                    elapsed,
                    elapsed,
                    0,
                    0,
                    &msg,
                )?;
            }
        }
    }

    let wall = overall.elapsed().as_millis();
    let ok_count = results.iter().filter(|r| r.4).count();
    println!();
    println!("# total: {ok_count}/{} ok · {wall}ms wall", results.len());
    println!();
    println!("| agreement | total ms | chars | sections |");
    println!("|---|---:|---:|---:|");
    for (slug, t, c, s, _) in &results {
        println!("| {slug} | {t} | {c} | {s} |");
    }

    // Aggregated rollup (mirrors the TS reporter).
    let mut totals: Vec<u128> = results.iter().filter(|r| r.4).map(|r| r.1).collect();
    let chars: Vec<u128> = results.iter().filter(|r| r.4).map(|r| r.2 as u128).collect();
    let sections: Vec<u128> = results.iter().filter(|r| r.4).map(|r| r.3 as u128).collect();
    if !totals.is_empty() {
        let sum_total: u128 = totals.iter().sum();
        let sum_chars: u128 = chars.iter().sum();
        let sum_sec: u128 = sections.iter().sum();
        let median = percentile(&mut totals.clone(), 0.5);
        let min = percentile(&mut totals.clone(), 0.0);
        let max = percentile(&mut totals.clone(), 1.0);
        let avg = sum_total / totals.len() as u128;
        let throughput = if sum_total > 0 {
            (sum_chars as f64 * 1000.0) / sum_total as f64
        } else {
            0.0
        };
        println!();
        println!(
            "# aggregated for backend={backend} model={model} (n={})",
            totals.len()
        );
        println!("| metric | min | median | avg | max | sum |");
        println!("|---|---:|---:|---:|---:|---:|");
        println!("| total_ms | {min} | {median} | {avg} | {max} | {sum_total} |");
        println!(
            "| chars    | {} | {} | {} | {} | {sum_chars} |",
            chars.iter().min().unwrap(),
            percentile(&mut chars.clone(), 0.5),
            sum_chars / chars.len() as u128,
            chars.iter().max().unwrap(),
        );
        println!(
            "| sections | {} | {} | {:.1} | {} | {sum_sec} |",
            sections.iter().min().unwrap(),
            percentile(&mut sections.clone(), 0.5),
            sum_sec as f64 / sections.len() as f64,
            sections.iter().max().unwrap(),
        );
        println!();
        println!(
            "# throughput: {sum_chars} chars in {:.1}s sequential = {:.0} chars/sec",
            sum_total as f64 / 1000.0,
            throughput
        );
    }

    if ok_count == results.len() {
        Ok(())
    } else {
        Err(format!("{}/{} failed", results.len() - ok_count, results.len()).into())
    }
}
