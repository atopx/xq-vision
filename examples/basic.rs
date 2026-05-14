use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use xq_vision::ModelSource;
use xq_vision::PieceKind;
use xq_vision::PieceRecognition;
use xq_vision::XqVision;

const TEST_IMAGE: &str = "examples/test.jpg";
const OUTPUT_HTML: &str = "examples/basic.html";
const PIECE_SIZE_MIN_PX: f32 = 28.0;
const PIECE_SIZE_VIEWPORT_VW: f32 = 5.8;
const PIECE_SIZE_MAX_PX: f32 = 44.0;
const PIECE_FONT_MIN_PX: f32 = 18.0;
const PIECE_FONT_VIEWPORT_VW: f32 = 3.3;
const PIECE_FONT_MAX_PX: f32 = 30.0;
const PIECE_EDGE_CLEARANCE_PX: f32 = 12.0;
const BOARD_GRID_INSET_PX: f32 = PIECE_SIZE_MAX_PX / 2.0 + PIECE_EDGE_CLEARANCE_PX;
const BOARD_GRID_DOUBLE_INSET_PX: f32 = BOARD_GRID_INSET_PX * 2.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let board_model = required_env("BOARD_MODEL")?;
    let piece_model = required_env("PIECE_MODEL")?;
    let image_path = manifest_dir.join(TEST_IMAGE);
    let html_path = manifest_dir.join(OUTPUT_HTML);

    let image = image::open(&image_path)?.to_rgb8();
    let mut vision = XqVision::builder()
        .board_model(ModelSource::file(board_model.clone()))
        .piece_model(ModelSource::file(piece_model.clone()))
        .build()?;

    let result = vision.recognize(&image)?;
    let html = render_html(&image_path, result.pieces());
    fs::write(&html_path, html)?;

    println!("wrote {}", html_path.display());
    println!("fen {}", result.pieces().to_fen_placement());

    Ok(())
}

fn required_env(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    env::var(name).map_err(|_| format!("missing required environment variable {name}").into())
}

fn render_html(image_path: &Path, pieces: &PieceRecognition) -> String {
    let pieces_html = render_pieces(pieces);
    let board_grid = render_board_grid();
    let image_src = escape_html(image_path.file_name().and_then(|name| name.to_str()).unwrap_or(TEST_IMAGE));

    format!(
        r#"<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>识别结果</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #f4f1ea;
      color: #211b15;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1100px, calc(100vw - 32px));
      margin: 16px auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 24px;
      align-items: start;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 16px;
      font-weight: 700;
    }}
    .panel {{
      background: #fffaf1;
      border: 1px solid #d7c8aa;
      border-radius: 8px;
      padding: 16px;
    }}
    img {{
      display: block;
      width: 100%;
      max-width: 480px;
      aspect-ratio: 8 / 9;
      object-fit: contain;
      margin: 0 auto;
      background: #eee2c8;
      border: 1px solid #d0c0a2;
      border-radius: 6px;
    }}
    .board {{
      position: relative;
      --piece-size: clamp({PIECE_SIZE_MIN_PX}px, {PIECE_SIZE_VIEWPORT_VW}vw, {PIECE_SIZE_MAX_PX}px);
      --piece-font-size: clamp({PIECE_FONT_MIN_PX}px, {PIECE_FONT_VIEWPORT_VW}vw, {PIECE_FONT_MAX_PX}px);
      --grid-inset: {BOARD_GRID_INSET_PX}px;
      width: 100%;
      max-width: 480px;
      margin: 0 auto;
      aspect-ratio: 8 / 9;
      background: #edd19a;
      border: 2px solid #8c5a21;
      border-radius: 8px;
      box-shadow: inset 0 0 0 2px #c98f42;
    }}
    .board-grid {{
      position: absolute;
      top: var(--grid-inset);
      left: var(--grid-inset);
      width: calc(100% - {BOARD_GRID_DOUBLE_INSET_PX}px);
      height: calc(100% - {BOARD_GRID_DOUBLE_INSET_PX}px);
      overflow: visible;
      pointer-events: none;
    }}
    .board-grid line {{
      stroke: #6a3d12;
      stroke-width: 1;
      fill: none;
      vector-effect: non-scaling-stroke;
    }}
    .board-grid .frame {{
      stroke-width: 1.6;
    }}
    .board-grid .river-text {{
      fill: #5a3812;
      font-family: "STKaiti", "KaiTi", "FangSong", "Songti SC", serif;
      font-size: 8px;
      font-weight: 700;
      letter-spacing: 3px;
      text-anchor: middle;
      dominant-baseline: middle;
    }}
    .piece {{
      position: absolute;
      left: var(--x);
      top: var(--y);
      transform: translate(-50%, -50%);
      z-index: 1;
      width: var(--piece-size);
      height: var(--piece-size);
      display: grid;
      place-items: center;
      border-radius: 50%;
      background:
        radial-gradient(circle at 35% 28%, #fff7e4 0, #f9ead0 42%, #d7b77b 100%);
      border: 2px solid #7f4f1d;
      font-size: var(--piece-font-size);
      font-weight: 800;
      line-height: 1;
      box-shadow: 0 2px 4px rgba(56, 37, 18, 0.28);
    }}
    .piece.red {{ color: #b51618; }}
    .piece.black {{ color: #17130f; }}
    .piece.unknown {{ color: #805d24; }}
    @media (max-width: 860px) {{
      main {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h2>原始图片</h2>
      <img src="{image_src}" alt="test image">
    </section>
    <section class="panel">
      <h2>识别后绘制的棋盘</h2>
      <div class="board" aria-label="recognized xiangqi board">
        {board_grid}
        {pieces_html}
      </div>
    </section>
  </main>
</body>
</html>
"#
    )
}

fn render_board_grid() -> String {
    // viewBox uses 10 units per cell so file/rank intersections land on integer
    // multiples of 10 — keeps every coordinate easy to read at a glance.
    let mut svg =
        String::from(r#"<svg class="board-grid" viewBox="0 0 80 90" preserveAspectRatio="none" aria-hidden="true">"#);

    for rank in 0..=9u32 {
        let y = rank * 10;
        svg.push_str(&format!(r#"<line x1="0" y1="{y}" x2="80" y2="{y}" />"#));
    }

    svg.push_str(r#"<line class="frame" x1="0" y1="0" x2="0" y2="90" />"#);
    svg.push_str(r#"<line class="frame" x1="80" y1="0" x2="80" y2="90" />"#);

    for file in 1..=7u32 {
        let x = file * 10;
        svg.push_str(&format!(r#"<line x1="{x}" y1="0" x2="{x}" y2="40" />"#));
        svg.push_str(&format!(r#"<line x1="{x}" y1="50" x2="{x}" y2="90" />"#));
    }

    // Palace diagonals — black (top) then red (bottom).
    svg.push_str(r#"<line x1="30" y1="0" x2="50" y2="20" />"#);
    svg.push_str(r#"<line x1="50" y1="0" x2="30" y2="20" />"#);
    svg.push_str(r#"<line x1="30" y1="70" x2="50" y2="90" />"#);
    svg.push_str(r#"<line x1="50" y1="70" x2="30" y2="90" />"#);

    svg.push_str(r#"<text x="20" y="46" class="river-text">楚 河</text>"#);
    svg.push_str(r#"<text x="60" y="46" class="river-text">汉 界</text>"#);

    svg.push_str("</svg>");
    svg
}

fn render_pieces(pieces: &PieceRecognition) -> String {
    let mut html = String::new();
    for row in pieces.cells() {
        for cell in row {
            if cell.piece == PieceKind::Empty {
                continue;
            }
            let title = format!(
                "rank {}, file {}, class {}, confidence {:.4}",
                cell.coord.rank,
                cell.coord.file,
                cell.piece.index(),
                cell.confidence
            );
            let class_name = match cell.piece {
                PieceKind::RedKing
                | PieceKind::RedAdvisor
                | PieceKind::RedBishop
                | PieceKind::RedKnight
                | PieceKind::RedRook
                | PieceKind::RedCannon
                | PieceKind::RedPawn => "red",
                PieceKind::Unknown => "unknown",
                _ => "black",
            };
            let (x, y) = intersection_position(cell.coord.file, cell.coord.rank);
            html.push_str(&format!(
                r#"<span class="piece {class_name}" style="--x: {x}; --y: {y};" title="{}">{}</span>"#,
                escape_html(&title),
                piece_label(cell.piece)
            ));
        }
    }
    html
}

fn intersection_position(file: usize, rank: usize) -> (String, String) {
    (axis_position(file as f32 / 8.0), axis_position(rank as f32 / 9.0))
}

fn axis_position(ratio: f32) -> String {
    let percent = ratio * 100.0;
    let pad_offset = BOARD_GRID_INSET_PX * (1.0 - 2.0 * ratio);
    if pad_offset >= 0.0 {
        format!("calc({percent:.6}% + {pad_offset:.3}px)")
    } else {
        format!("calc({percent:.6}% - {:.3}px)", pad_offset.abs())
    }
}

fn piece_label(piece: PieceKind) -> &'static str {
    match piece {
        PieceKind::Empty => "",
        PieceKind::Unknown => "?",
        PieceKind::RedKing => "帅",
        PieceKind::RedAdvisor => "仕",
        PieceKind::RedBishop => "相",
        PieceKind::RedKnight => "马",
        PieceKind::RedRook => "车",
        PieceKind::RedCannon => "炮",
        PieceKind::RedPawn => "兵",
        PieceKind::BlackKing => "将",
        PieceKind::BlackAdvisor => "士",
        PieceKind::BlackBishop => "象",
        PieceKind::BlackKnight => "马",
        PieceKind::BlackRook => "车",
        PieceKind::BlackCannon => "炮",
        PieceKind::BlackPawn => "卒",
    }
}

fn escape_html(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}
