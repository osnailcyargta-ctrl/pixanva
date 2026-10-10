// E2E prompter — mirror 1:1 logika prompterParse di worker.js
import { fetchModel, Pixanva } from "../js/engine.js";
import fs from "fs";

const itos = JSON.parse(fs.readFileSync("models/prompt_vocab.json", "utf8"));
const stoi = {};
itos.itos.forEach((w, i) => { if (w) stoi[w] = i; });
const S = itos.specials;

const { meta, W } = await fetchModel("http://localhost:8901/models/prompter.bin", () => {});
const model = new Pixanva(meta, W);
console.log("loaded:", meta.name, meta.stage, meta.n_params.toLocaleString(), "params, step", meta.step);

function tok(s) { return s.toLowerCase().match(/[a-z0-9]+/g) || []; }
function parse(text) {
  const ids = [S.BOS];
  for (const w of tok(text)) if (stoi[w] !== undefined) ids.push(stoi[w]);
  ids.push(S.A);
  const st = model.newState(ids.length + 8);
  let pos = 0;
  for (const t of ids) { model.step(st, t, pos, 0); pos++; }
  const out = [];
  for (let i = 0; i < 5; i++) {
    const lg = st.logits;
    let best = 0, bv = -1e30;
    for (let v = 0; v < model.V; v++) if (lg[v] > bv) { bv = lg[v]; best = v; }
    out.push(itos.itos[best] || "");
    if (i < 4) { model.step(st, best, pos, 0); pos++; }
  }
  const tag = (w) => (w && w[0] === "@" && w !== "@none") ? w.slice(1) : null;
  return { scene: tag(out[0]), color: tag(out[1]), mood: tag(out[2]),
           orn: [tag(out[3]), tag(out[4])].filter(Boolean) };
}

const tests = [
  "gw mau pantai senja ada perahu dong",
  "gunung salju pagi ada pohon cemara",
  "kota malam neon berbintang",
  "bikinin danau berkabut warna pastel",
  "air terjun tropis siang ada pelangi",
  "aurora malam es ada meteor bro",
  "pantai senja ada perahu",          // tanpa kata mubazir
  "gurun yang hangat gitu loh",       // mood none di tengah
  "ladang bunga pastel with kupu kupu", // english mix
];
for (const t of tests) {
  const r = parse(t);
  console.log(`"${t}"  ->  [${r.scene}, ${r.color}, ${r.mood}, (${r.orn.join(", ") || "-"})]`);
}
