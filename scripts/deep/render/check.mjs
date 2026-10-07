// node check.mjs <mermaid|mermaid10|tables> FILE...
//
// mermaid / mermaid10: each .mmd file parses with Mermaid 11 / 10.
// tables: each .md file's Columns table parses with GFM rules (markdown-it)
// to one row per column, every row with the header's nine cells, each first
// cell the column's name as `NAME.names` lists them (one per line).
//
// Prints "ok   FILE" or "FAIL FILE: why" per file; exits 1 on any failure.
import fs from "node:fs";
import { JSDOM } from "jsdom";

const [mode, ...files] = process.argv.slice(2);
let failed = 0;
const fail = (file, why) => {
  failed++;
  console.log(`FAIL ${file}: ${why}`);
};

if (mode === "mermaid" || mode === "mermaid10") {
  const dom = new JSDOM("<!doctype html><html><body></body></html>");
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  const { default: mermaid } = await import(mode);
  mermaid.initialize({ startOnLoad: false });
  for (const file of files) {
    try {
      await mermaid.parse(fs.readFileSync(file, "utf8"));
      console.log(`ok   ${file}`);
    } catch (e) {
      fail(file, String(e.message || e).split("\n").slice(0, 3).join(" / "));
    }
  }
} else if (mode === "tables") {
  const { default: MarkdownIt } = await import("markdown-it");
  const md = new MarkdownIt();
  for (const file of files) {
    const names = fs.readFileSync(file.replace(/\.md$/, ".names"), "utf8").split("\n");
    const rows = [];
    let cells = null, cell = null, inTable = false, done = false;
    for (const t of md.parse(fs.readFileSync(file, "utf8"), {})) {
      if (done) break;
      if (t.type === "table_open") inTable = true;
      else if (t.type === "table_close") done = true;
      else if (!inTable) continue;
      else if (t.type === "tr_open") cells = [];
      else if (t.type === "tr_close") rows.push(cells);
      else if (t.type === "th_open" || t.type === "td_open") cell = "";
      else if (t.type === "th_close" || t.type === "td_close") cells.push(cell);
      else if (t.type === "inline") cell = t.children.map((c) => c.content).join("");
    }
    const body = rows.slice(1);
    if (!rows.length) fail(file, "no table");
    else if (rows.some((r) => r.length !== 9)) fail(file, "a row without nine cells");
    else if (body.length !== names.length) fail(file, `${body.length} rows for ${names.length} columns`);
    else {
      // A backslash before a pipe is doubled for GitHub, which markdown-it
      // shows doubled: the one name it cannot round-trip exactly.
      const bad = body.findIndex(
        (r, i) => r[0].trim() !== names[i].trim() && !names[i].includes("\\")
      );
      if (bad >= 0) fail(file, `${JSON.stringify(body[bad][0])} != ${JSON.stringify(names[bad])}`);
      else console.log(`ok   ${file}`);
    }
  }
} else {
  console.error("usage: node check.mjs <mermaid|mermaid10|tables> FILE...");
  process.exit(2);
}
process.exit(failed ? 1 : 0);
