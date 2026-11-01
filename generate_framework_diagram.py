"""
EvoRAG System Architecture Generator (Publication-Grade Academic Standard)
==========================================================================

Generates an exact publication-grade academic system architecture diagram:
  - Pure White Background (#FFFFFF)
  - Solid Black Outline Boxes & Containers (#000000)
  - Times New Roman Typography throughout
  - Perfect geometry, alignment, arrow padding, and zero visual clutter

Output: evorag_design_framework.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_standard_academic_architecture():
    # Set font to Times New Roman globally
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

    # Set portrait canvas size (15 x 22 inches @ 300 DPI)
    fig, ax = plt.subplots(figsize=(15, 22), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')  # Pure white background
    ax.set_facecolor('#FFFFFF')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Style Constants
    COLOR_BORDER = '#000000'
    COLOR_BG = '#FFFFFF'
    COLOR_TEXT = '#000000'

    # Helper function to draw container bounding box with top title
    def draw_container(x, y, w, h, title):
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1.6, edgecolor=COLOR_BORDER, facecolor='none', zorder=1
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.2, title, ha='center', va='center', fontsize=16, fontweight='bold', color=COLOR_TEXT, fontfamily='Times New Roman', zorder=2)

    # Helper function to draw standard node box
    def draw_box(x, y, w, h, lines, fontsize=12.5, fontweight='normal', bold_first=False):
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1.3, edgecolor=COLOR_BORDER, facecolor=COLOR_BG, zorder=3
        )
        ax.add_patch(rect)

        num_lines = len(lines)
        if num_lines == 1:
            ax.text(x + w/2, y + h/2, lines[0], ha='center', va='center', fontsize=fontsize, fontweight=fontweight, color=COLOR_TEXT, fontfamily='Times New Roman', zorder=4)
        else:
            line_height = (h - 0.8) / max(num_lines, 1)
            start_y = y + h - 0.6 - line_height/2
            for idx, line in enumerate(lines):
                fw = 'bold' if (idx == 0 and bold_first) else fontweight
                ax.text(x + w/2, start_y - (idx * line_height), line, ha='center', va='center', fontsize=fontsize, fontweight=fw, color=COLOR_TEXT, fontfamily='Times New Roman', zorder=4)

    # Helper function to draw arrows
    def draw_arrow(x1, y1, x2, y2, label="", linestyle='-', rad=0.0):
        connectionstyle = f"arc3,rad={rad}" if rad != 0.0 else None
        arrow_props = dict(arrowstyle="-|>", color=COLOR_BORDER, lw=1.5, mutation_scale=14, ls=linestyle)
        if connectionstyle:
            arrow_props['connectionstyle'] = connectionstyle

        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=arrow_props, zorder=5
        )
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 1.0, label, ha='center', va='center', fontsize=11, color=COLOR_TEXT, fontfamily='Times New Roman', zorder=6)

    # -------------------------------------------------------------------------
    # TITLE BANNER
    # -------------------------------------------------------------------------
    ax.text(50, 98.2, "EvoRAG System Architecture", 
            ha='center', va='center', fontsize=26, fontweight='bold', color=COLOR_TEXT, fontfamily='Times New Roman')

    # -------------------------------------------------------------------------
    # 1. DATA LAYER (Container)
    # -------------------------------------------------------------------------
    draw_container(10, 74, 80, 20, "Data Layer")

    draw_box(36, 88.5, 28, 3.8, ["News Data Corpus"], fontsize=13, fontweight='bold')
    draw_box(34, 82.5, 32, 3.8, ["Dataset Partitioning & Chunking"], fontsize=12.5)
    
    draw_box(16, 75.8, 30, 3.8, ["Retrieved Context Chunks"], fontsize=12.5)
    draw_box(54, 75.8, 30, 3.8, ["Evaluation Query Set"], fontsize=12.5)

    # Arrows inside Data Layer
    draw_arrow(50, 88.5, 50, 86.3)
    draw_arrow(50, 82.5, 31, 79.6)
    draw_arrow(50, 82.5, 69, 79.6)

    # -------------------------------------------------------------------------
    # 2. PERSONA BRANCHING LAYER
    # -------------------------------------------------------------------------
    p_y = 65.2
    p_h = 4.0
    p_w = 14.0

    draw_box(10, p_y, p_w, p_h, ["Persona 1"], fontsize=12, fontweight='bold')
    draw_box(28, p_y, p_w, p_h, ["Persona 2"], fontsize=12, fontweight='bold')
    draw_box(46, p_y, p_w, p_h, ["Persona 3"], fontsize=12, fontweight='bold')
    ax.text(65, p_y + p_h/2, "•  •  •  •  •  •", ha='center', va='center', fontsize=14, fontweight='bold', color=COLOR_TEXT)
    draw_box(76, p_y, p_w, p_h, ["Persona N"], fontsize=12, fontweight='bold')

    # Arrows from Retrieved Context Chunks to Personas
    draw_arrow(31, 75.8, 17, 69.2)
    draw_arrow(31, 75.8, 35, 69.2)
    draw_arrow(31, 75.8, 53, 69.2)
    draw_arrow(31, 75.8, 83, 69.2)

    # -------------------------------------------------------------------------
    # 3. MULTI-PERSONA ENGINE (Container)
    # -------------------------------------------------------------------------
    draw_container(10, 52.0, 80, 9.5, "Multi-Persona Engine")

    draw_box(14, 53.5, 20, 4.2, ["Context Grounding"], fontsize=12)
    draw_box(40, 53.5, 20, 4.2, ["Parallel LLM Generation"], fontsize=12)
    draw_box(66, 53.5, 20, 4.2, ["Perspective Outputs"], fontsize=12)

    draw_arrow(34, 55.6, 40, 55.6)
    draw_arrow(60, 55.6, 66, 55.6)

    # Arrows from Personas down into Multi-Persona Engine
    draw_arrow(17, p_y, 24, 61.5)
    draw_arrow(35, p_y, 24, 61.5)
    draw_arrow(53, p_y, 50, 61.5)
    draw_arrow(83, p_y, 76, 61.5)

    # -------------------------------------------------------------------------
    # 4. CONSENSUS & VOTING STAGE
    # -------------------------------------------------------------------------
    draw_arrow(50, 52.0, 30, 46.8)
    draw_arrow(50, 52.0, 70, 46.8)

    draw_box(16, 42.5, 28, 4.3, ["Peer Evaluation Matrix"], fontsize=12.5)
    draw_box(56, 42.0, 28, 4.8, ["Aggregate Functions", "*Grounding  *Reasoning  *Completeness"], fontsize=11.5)

    draw_arrow(30, 42.5, 50, 39.2)
    draw_arrow(70, 42.0, 50, 39.2)

    draw_box(32, 35.0, 36, 4.2, ["Robust Consensus Synthesis"], fontsize=13, fontweight='bold')

    # -------------------------------------------------------------------------
    # 5. FINAL ANSWER STAGE
    # -------------------------------------------------------------------------
    draw_arrow(50, 35.0, 50, 31.4)
    draw_box(30, 27.2, 40, 4.2, ["Synthesized Final Answer & Citations"], fontsize=13, fontweight='bold')

    # -------------------------------------------------------------------------
    # 6. EVOLUTIONARY ADAPTATION LAYER (Container)
    # -------------------------------------------------------------------------
    draw_arrow(50, 27.2, 50, 23.2)
    draw_container(20, 5.5, 60, 17.7, "Evaluation & Evolution Layer")

    draw_box(24, 7.2, 52, 13.0, [
        "Evaluation Metrics & Evolution",
        "1. Historical Score Logging",
        "2. Rolling Window Tracker (N = 5)",
        "3. Weak Persona Detector",
        "4. Prompt Mutation & Persona Hot-Swap"
    ], fontsize=12.5, bold_first=True)

    # Side Feedback Loop Arrow (Evolution Layer -> Multi-Persona Engine)
    draw_arrow(80, 14.35, 90, 56.75, linestyle='--', rad=-0.4)
    ax.text(93.0, 35.5, "Evolutionary Feedback Loop\n(Persona Hot-Swap)", ha='center', va='center', fontsize=11, color=COLOR_TEXT, fontfamily='Times New Roman', zorder=6)

    # Save output image
    output_filename = "evorag_design_framework.png"
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    print(f"[+] Publication-Grade Academic Architecture Diagram saved to: {output_filename}")

if __name__ == "__main__":
    generate_standard_academic_architecture()
