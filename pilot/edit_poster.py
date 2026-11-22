"""Rewrite slide1.xml of the poster: Methods pipeline up top, 'Design choices' cards below."""
import re
import shutil
import sys
from xml.sax.saxutils import escape

D = sys.argv[1]  # unpacked dir
FIG = sys.argv[2]
S = f"{D}/ppt/slides/slide1.xml"
x = open(S, encoding="utf-8").read()
CM = 360000


def shape(name):
    m = re.search(r'<p:(sp|pic)><p:nv(?:Sp|Pic)Pr><p:cNvPr id="\d+" name="' + re.escape(name) + r'".*?</p:\1>', x, re.S)
    assert m, name
    return m


def remove(name):
    global x
    m = shape(name)
    x = x[:m.start()] + x[m.end():]


def set_xfrm(s, off=None, ext=None):
    if off:
        s = re.sub(r'<a:off x="\d+" y="\d+"/>', f'<a:off x="{off[0]}" y="{off[1]}"/>', s, count=1)
    if ext:
        s = re.sub(r'<a:ext cx="\d+" cy="\d+"/>', f'<a:ext cx="{ext[0]}" cy="{ext[1]}"/>', s, count=1)
    return s


def replace_shape(name, new):
    global x
    m = shape(name)
    x = x[:m.start()] + new + x[m.end():]


def renamed(s, new_id, new_name):
    s = re.sub(r'<p:cNvPr id="\d+" name="[^"]*"', f'<p:cNvPr id="{new_id}" name="{new_name}"', s, count=1)
    return re.sub(r'<a:extLst><a:ext uri="\{FF2B5EF4[^}]*\}">.*?</a:ext></a:extLst>', "", s, count=1, flags=re.S)


def set_single_text(s, text):
    return re.sub(r"<a:t>[^<]*</a:t>", f"<a:t>{escape(text)}</a:t>", s, count=1)


FONT = ('<a:latin typeface="{f}" panose="{p}" pitchFamily="34" charset="0"/>'
        '<a:cs typeface="Segoe UI" panose="020B0502040204020203" pitchFamily="34" charset="0"/>')
SEGOE = FONT.format(f="Segoe UI", p="020B0502040204020203")
BLACK = FONT.format(f="Segoe UI Black", p="020B0A02040204020203")


def run(text, sz, bold_font=False, color=None, italic=False):
    fill = f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>' if color else ""
    it = ' i="1"' if italic else ""
    return (f'<a:r><a:rPr lang="en-US" sz="{sz}"{it} dirty="0">{fill}{BLACK if bold_font else SEGOE}</a:rPr>'
            f'<a:t xml:space="preserve">{escape(text)}</a:t></a:r>')


def textbox(sid, name, xy, wh, paras, anchor="t"):
    ps = "".join(f'<a:p><a:pPr><a:lnSpc><a:spcPct val="110000"/></a:lnSpc><a:spcAft><a:spcPts val="{aft}"/></a:spcAft>'
                 f'</a:pPr>{runs}</a:p>' for runs, aft in paras)
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{name}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{xy[0]}" y="{xy[1]}"/><a:ext cx="{wh[0]}" cy="{wh[1]}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="{anchor}">'
            f'<a:normAutofit/></a:bodyPr><a:lstStyle/>{ps}</p:txBody></p:sp>')


# ------------------------------------------------------------------ remove template results + old methods art
for n in ["Graphic 19", "Graphic 26", "TextBox 31", "TextBox 32", "TextBox 37", "TextBox 28"]:
    remove(n)
badge_tpl = shape("Rectangle: Rounded Corners 30").group(0)
remove("Rectangle: Rounded Corners 30")
remove("Rectangle: Rounded Corners 34")
card_tpl = shape("Rectangle 12").group(0)
heading_tpl = shape("TextBox 38").group(0)
tag_orange = shape("TextBox 4").group(0)
tag_purple = shape("TextBox 6").group(0)

X0, WTOT = 1781107, 26908262
TOP = 15163950  # where the 'Result 1/2' captions used to start (42.1 cm)

# ------------------------------------------------------------------ Methods block (moved up)
replace_shape("TextBox 38", set_xfrm(heading_tpl, off=(2007000, TOP)))
tag1 = set_single_text(set_xfrm(tag_orange, off=(6728568, TOP + 80774), ext=(int(13.6 * CM), 1096122)),
                       "50 paired problems")
tag2 = set_single_text(set_xfrm(tag_purple, off=(6728568 + int(14.4 * CM), TOP + 80774), ext=(int(11.0 * CM), 1096122)),
                       "Embedding-based")
replace_shape("TextBox 4", tag1)
replace_shape("TextBox 6", tag2)

CARD_Y, CARD_H = TOP + int(5.4 * CM), int(20.6 * CM)
replace_shape("Rectangle 35", set_xfrm(shape("Rectangle 35").group(0), off=(X0, CARD_Y), ext=(WTOT, CARD_H)))
fig_w = WTOT - int(2.0 * CM)
fig_h = round(fig_w * 1464 / 5724)
pic = shape("Picture 36").group(0)
pic = set_xfrm(pic, off=(X0 + int(1.0 * CM), CARD_Y + (CARD_H - fig_h) // 2), ext=(fig_w, fig_h))
pic = pic.replace('r:embed="rId5"', 'r:embed="rId7"')
pic = re.sub(r'descr="[^"]*"', 'descr="Pipeline: data, split into steps, codebook, label, count switches, compare"', pic)
replace_shape("Picture 36", pic)

# ------------------------------------------------------------------ Design choices block
DC_TOP = 25499618 - int(1.2 * CM)  # just under the Methods block (69.6 cm)
dc_head = set_single_text(renamed(set_xfrm(heading_tpl, off=(2007000, DC_TOP), ext=(int(27.5 * CM), 1256947)),
                                  201, "Design choices heading"), "Design choices")
dc_tag = set_single_text(renamed(set_xfrm(tag_orange, off=(2007000 + int(28.3 * CM), DC_TOP + 80774),
                                          ext=(int(15.5 * CM), 1096122)), 202, "Push back tag"), "Please push back!")
dc_tag = dc_tag.replace('val="F68B1F"', 'val="C0392B"').replace(
    '<a:rPr lang="en-US" sz="3600" dirty="0">', '<a:rPr lang="en-US" sz="3600" dirty="0"><a:solidFill><a:schemeClr val="bg1"/></a:solidFill>', 1)

C, P = "plain", "push"
CARDS = [
    ("What is a step?",
     [("Choice: ", "Line breaks; equations and short lines are glued on.", C),
      ("Push back: ", "QwQ gets ~2× more, shorter steps. Use sentences instead?", P)]),
    ("Distance or Vendi?",
     [("A: ", "cosine distance between a step and the previous one (a local jump).", C),
      ("B: ", "Vendi gain: how much a step adds to the diversity of all steps so far.", C),
      ("Push back: ", "Which one captures exploring better?", P)]),
    ("What about loops?",
     [("Choice: ", "A far jump back to an earlier step counts as a loop, not as new exploration.", C),
      ("Push back: ", "Is going back exploring, exploiting, or its own category?", P)]),
    ("Miracle steps?",
     [("Choice: ", "A big jump in the last 20% of a solution is a candidate.", C),
      ("Push back: ", "Most candidates are checks (“which matches”). Add a correctness check?", P)]),
    ("Strategy labels?",
     [("Instead of explore/exploit: ", "an LLM names 3–6 strategies per problem (e.g. “substitute x = 0”, "
       "“parity argument”) and labels every step; a switch is a change of label.", C),
      ("Push back: ", "Is this a viable replacement?", P)]),
    ("Fair comparison?",
     [("Choice: ", "Human stage-1 write-ups vs. QwQ’s raw trace, paired by problem (n = 50).", C),
      ("Push back: ", "Humans wrote after solving. Are their dead ends missing?", P)]),
]
GAP = int(1.2 * CM)
NCOL = 3
CW = (WTOT - (NCOL - 1) * GAP) // NCOL
CH = int(12.2 * CM)
ROW_Y = [DC_TOP + int(4.6 * CM), DC_TOP + int(4.6 * CM) + CH + int(1.0 * CM)]
new = [dc_head, dc_tag]
sid = 210
for i, (title, paras) in enumerate(CARDS):
    cx, cy = X0 + (i % NCOL) * (CW + GAP), ROW_Y[i // NCOL]
    if i == 0:
        replace_shape("Rectangle 16", set_xfrm(shape("Rectangle 16").group(0), off=(cx, cy), ext=(CW, CH)))
    elif i == 1:
        replace_shape("Rectangle 12", set_xfrm(shape("Rectangle 12").group(0), off=(cx, cy), ext=(CW, CH)))
    else:
        new.append(renamed(set_xfrm(card_tpl, off=(cx, cy), ext=(CW, CH)), sid, f"Card {i + 1}"))
    badge = renamed(set_xfrm(badge_tpl, off=(cx + int(1.0 * CM), cy + int(1.0 * CM))), sid + 1, f"Card {i + 1} number")
    new.append(set_single_text(badge, str(i + 1)))
    new.append(textbox(sid + 2, f"Card {i + 1} title", (cx + int(4.2 * CM), cy + int(1.0 * CM)),
                       (CW - int(5.0 * CM), int(2.54 * CM)), [(run(title, 3800, bold_font=True), 0)], anchor="ctr"))
    new.append(textbox(sid + 3, f"Card {i + 1} text", (cx + int(1.0 * CM), cy + int(4.1 * CM)),
                       (CW - int(2.0 * CM), CH - int(4.7 * CM)),
                       [(run(lab, 2800, bold_font=True, color="C0392B" if kind == P else None)
                         + run(txt, 2800, italic=kind == P), 900 if j < len(paras) - 1 else 0)
                        for j, (lab, txt, kind) in enumerate(paras)]))
    sid += 4

# ------------------------------------------------------------------ Limitation box (replace template text)
OLD_LIM = ("This is only one interpretation of findings that suggest effective posters are uncluttered, have big figures "
           "&amp; text, and include callout boxes with takeaways. It is missing your personality and creative flair.$")
assert OLD_LIM in x
x = x.replace(OLD_LIM, escape("This is a pilot with only 50 human and 50 LLM solutions. More LLM solutions could be "
                              "generated with prompt engineering, adjusted weights (fine-tuning) or deleted weights "
                              "(pruning) to specialise the model in reasoning. MiniLM was not trained on LaTeX."))
lim = shape("TextBox 43").group(0)
replace_shape("TextBox 43", set_xfrm(lim, off=(re.search(r'<a:off x="(\d+)"', lim).group(1), int(100.4 * CM)), ext=(re.search(r'<a:ext cx="(\d+)"', lim).group(1), int(9.2 * CM))))

x = x.replace("</p:spTree>", "".join(new) + "</p:spTree>")
open(S, "w", encoding="utf-8").write(x)

# picture relationship + media
shutil.copy(FIG, f"{D}/ppt/media/image6.png")
R = f"{D}/ppt/slides/_rels/slide1.xml.rels"
r = open(R).read()
r = r.replace("</Relationships>", '<Relationship Id="rId7" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
              'relationships/image" Target="../media/image6.png"/></Relationships>')
for rid in ("rId3", "rId4", "rId5"):
    r = re.sub(rf'<Relationship Id="{rid}"[^>]*/>', "", r)
open(R, "w").write(r)
print("ok")
