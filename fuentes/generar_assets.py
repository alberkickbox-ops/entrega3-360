"""
Genera los recursos 360º del proyecto (obra original, sin recursos de terceros):
  - assets/planeta-lejano.jpg  (panorámica equirrectangular 4096x2048)
  - assets/aurora.mp4          (vídeo equirrectangular 2048x1024, bucle de 10 s)
Requiere: numpy, pillow y ffmpeg.   Uso: python3 generar_assets.py
"""
import subprocess, numpy as np
from PIL import Image

rng = np.random.default_rng(7)

def smooth(x): return x * x * (3 - 2 * x)

def directions(W, H, top_only=False):
    """Vector 3D unitario para cada píxel de una equirrectangular (ruido sin costuras)."""
    rows = H // 2 if top_only else H
    u = (np.arange(W) + 0.5) / W * 2 * np.pi
    v = np.pi / 2 - (np.arange(rows) + 0.5) / H * np.pi
    U, V = np.meshgrid(u, v)
    return np.cos(V) * np.cos(U), np.sin(V), np.cos(V) * np.sin(U), U, V

def nebula(x, y, z, seed):
    r = np.random.default_rng(seed)
    f = np.zeros_like(x)
    for i in range(7):
        k = r.normal(size=3); k /= np.linalg.norm(k)
        freq = 2.0 + i * 1.7
        f += np.sin(freq * (k[0]*x + k[1]*y + k[2]*z) + r.uniform(0, 6.28)) / (1 + i * 0.6)
    return f

def add_stars(img, n, W, H, rows):
    for _ in range(n):
        u = rng.uniform(0, W)
        sv = rng.uniform(0, 1)            # distribución uniforme en la esfera
        v = np.arcsin(sv)
        py = (np.pi / 2 - v) / np.pi * H
        if py >= rows - 1: continue
        b = rng.uniform(0.3, 1.0) ** 2
        col = np.array(rng.choice([[1, 1, 1], [.8, .9, 1], [1, .9, .7]])) * b
        sx = min(1 / max(np.cos(v), .12), 8) * rng.uniform(.8, 1.4)
        for dx in range(-int(3 * sx) - 1, int(3 * sx) + 2):
            for dy in range(-3, 4):
                w = np.exp(-((dx / sx) ** 2 + dy ** 2) / 1.2)
                xx, yy = int(u + dx) % W, int(py + dy)
                if 0 <= yy < rows:
                    img[yy, xx] = np.minimum(1, img[yy, xx] + col * w)

def mirror_lake(top, W, H, t_phase=0.0):
    """Rellena la mitad inferior con el reflejo del cielo sobre un lago con ondas."""
    rows = top.shape[0]
    flip = top[::-1].copy()
    yy = np.arange(rows)[:, None]
    xs = np.arange(W)[None, :]
    shift = (np.sin(yy / 7.0 + t_phase) * (2 + yy / 40.0)).astype(int)
    idx = (xs + shift) % W
    lake = np.take_along_axis(flip, idx[..., None].repeat(3, 2), axis=1)
    fade = np.linspace(0.75, 0.25, rows)[:, None, None]
    lake = lake * fade * np.array([0.8, 0.95, 1.1])
    return np.concatenate([top, np.clip(lake, 0, 1)], axis=0)

def mountains(U, rows, H, amp, base, seed):
    r = np.random.default_rng(seed)
    h = np.zeros(U.shape[1])
    u1 = U[0]
    for k, a in [(3, 1.0), (5, .6), (9, .35), (17, .2), (31, .1), (63, .05)]:
        h += a * np.sin(k * u1 + r.uniform(0, 6.28))
    h = base + amp * (h - h.min()) / (h.max() - h.min())
    ys = np.arange(rows)[:, None]
    return (ys > (rows - h[None, :])).astype(np.float32)

# ---------------- IMAGEN: planeta lejano ----------------
def build_image(W=4096, H=2048):
    x, y, z, U, V = directions(W, H, top_only=True)
    rows = H // 2
    base = np.stack([0.01 + 0.04 * (1 - np.sin(V)) ** 2,
                     0.02 + 0.05 * (1 - np.sin(V)) ** 2,
                     0.07 + 0.12 * (1 - np.sin(V)) ** 2], -1)
    n1 = nebula(x, y, z, 11); n2 = nebula(x, y, z, 23)
    neb = np.clip(smooth(np.clip((n1 + 1.2) / 3.0, 0, 1)), 0, 1) ** 2.2
    neb2 = np.clip(smooth(np.clip((n2 + 1.0) / 3.0, 0, 1)), 0, 1) ** 2.5
    img = base + neb[..., None] * np.array([.55, .12, .45]) + neb2[..., None] * np.array([.05, .35, .5])
    # banda galáctica
    n = np.array([.3, .8, .5]); n /= np.linalg.norm(n)
    band = np.exp(-((x*n[0] + y*n[1] + z*n[2]) / 0.18) ** 2) * (0.5 + 0.5 * np.sin(14 * x + 8 * z))
    img += band[..., None] * np.array([.35, .3, .4]) * 0.8
    add_stars(img, 2600, W, H, rows)
    # planeta con anillos
    p0 = np.array([.55, .22, -.8]); p0 /= np.linalg.norm(p0)
    e1 = np.cross(p0, [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(p0, e1)
    px, py_, pz = x*e1[0]+y*e1[1]+z*e1[2], x*e2[0]+y*e2[1]+z*e2[2], x*p0[0]+y*p0[1]+z*p0[2]
    R = 0.30
    front = pz > 0
    rr = np.sqrt(px**2 + py_**2)
    disc = front & (rr < R)
    nz = np.sqrt(np.clip(1 - (rr / R) ** 2, 0, 1))
    lx, ly = -0.6, 0.5
    light = np.clip((px / R) * lx + (py_ / R) * ly + nz * 0.7, 0, 1)
    bands = 0.5 + 0.5 * np.sin(py_ / R * 14 + 2 * np.sin(px / R * 3))
    col = (np.array([.85, .45, .25]) * bands[..., None] + np.array([.95, .8, .55]) * (1 - bands[..., None])) * (0.15 + light[..., None])
    # anillo (elipse inclinada)
    ang = np.deg2rad(-18)
    rx = px * np.cos(ang) - py_ * np.sin(ang); ry = px * np.sin(ang) + py_ * np.cos(ang)
    ring_r = np.sqrt((rx / 1.0) ** 2 + (ry / 0.28) ** 2) / R
    ring = front & (ring_r > 1.25) & (ring_r < 2.1)
    ring_a = (0.55 + 0.45 * np.sin(ring_r * 60)) * np.exp(-((ring_r - 1.7) / 0.5) ** 2)
    ring_col = np.array([.9, .85, .75])[None, None, :] * ring_a[..., None]
    ring_front = ry < 0
    # anillo detrás del disco, delante del disco si ry<0
    img = np.where((ring & ~disc)[..., None], img * 0.3 + ring_col * 0.9, img)
    img = np.where(disc[..., None], col, img)
    img = np.where((ring & ring_front & disc)[..., None], col * 0.2 + ring_col, img)
    glow = np.exp(-(np.clip(rr - R, 0, None) / 0.05) ** 2) * front
    img += glow[..., None] * np.array([.4, .25, .15]) * 0.5
    # montañas
    m = mountains(U, rows, H, amp=150, base=30, seed=5)
    mt_col = np.array([.02, .025, .05]) + 0.06 * np.linspace(0, 1, rows)[:, None, None] * np.array([1, .5, .8])
    img = np.where(m[..., None] > 0, np.broadcast_to(mt_col, img.shape), img)
    img = np.clip(img, 0, 1)
    full = mirror_lake(img, W, H)
    Image.fromarray((full ** (1 / 1.1) * 255).astype(np.uint8)).save(
        "../assets/planeta-lejano.jpg", quality=88, optimize=True)

# ---------------- VÍDEO: aurora boreal ----------------
def build_video(W=2048, H=1024, fps=24, secs=10):
    x, y, z, U, V = directions(W, H, top_only=True)
    rows = H // 2
    frames = fps * secs
    sky = np.stack([0.005 + 0.02 * (1 - np.sin(V)) ** 3,
                    0.01 + 0.04 * (1 - np.sin(V)) ** 3,
                    0.03 + 0.09 * (1 - np.sin(V)) ** 3], -1)
    stars = np.zeros((rows, W, 3), np.float32); add_stars(stars, 1500, W, H, rows)
    twink = rng.uniform(0, 6.28, (rows, W, 1)).astype(np.float32)
    m = mountains(U, rows, H, amp=70, base=14, seed=9)
    ys = np.arange(rows)[:, None]
    mt = np.broadcast_to(np.array([.01, .015, .02]), (rows, W, 3))
    # copas de abetos en el horizonte
    tree = ((np.sin(U * 220) > 0.2) & (ys > rows - 24 - 10 * np.abs(np.sin(U * 90))[0][None, :] * 0 - 12 * (np.abs(np.sin(U * 130)) )))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "27", "-preset", "slow", "-movflags", "+faststart", "-an", "../assets/aurora.mp4"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(frames):
        t = 2 * np.pi * f / frames        # bucle perfecto: sólo armónicos enteros
        img = sky + stars * (0.65 + 0.35 * np.sin(twink + 3 * t))
        total = np.zeros_like(img)
        for (c0, a1, a2, w0, hue) in [(.42, .12, .07, .13, 0), (.70, .10, .05, .09, 1)]:
            centre = c0 + a1 * np.sin(2 * U + t) + a2 * np.sin(5 * U - 2 * t + hue)
            w = w0 * (0.8 + 0.3 * np.sin(3 * U + t + hue))
            d = (V - centre) / w
            cur = np.exp(-d ** 2) * (d < 0) + np.exp(-(d * 2.2) ** 2) * (d >= 0)
            rays = 0.55 + 0.45 * np.sin(55 * U + 4 * np.sin(3 * U + t) + 2 * t) ** 2
            inten = cur * rays * (0.6 + 0.4 * np.sin(4 * U - t + hue * 2))
            g = np.clip((V - centre) / w, -1, 2) * 0.5 + 0.5
            c = np.stack([0.15 + 0.7 * g ** 2, 0.95 - 0.35 * g, 0.45 + 0.5 * g], -1)
            if hue: c = c[..., [2, 1, 0]] * np.array([1.0, .9, 1.0])
            total += inten[..., None] * c
        img = img + total * 0.9
        img = np.where(m[..., None] > 0, mt + total * 0.02, img)
        img = np.clip(img, 0, 1)
        full = mirror_lake(img, W, H, t_phase=2 * t)
        p.stdin.write((full ** (1 / 1.15) * 255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()

if __name__ == "__main__":
    build_image(); build_video()
