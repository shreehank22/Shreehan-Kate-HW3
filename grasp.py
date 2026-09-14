import numpy as np
import matplotlib.pyplot as plt

A, B = 3.0, 1.5   

def skew_symmetric_matrix(a):
    ax, ay, az = a
    return np.array([[0, -az, ay],[az, 0, -ax],[-ay, ax, 0]])

def P_matrix(o,c_i):
    r = np.array(c_i)-np.array(o)
    top = np.hstack([np.eye(3), skew_symmetric_matrix(r).T])
    bot = np.hstack([np.zeros((3, 3)), np.eye(3)])
    return np.vstack([top,bot])

def R_matrix(n_hat, t_hat, m_hat):
    R_Ci_N = np.column_stack([n_hat, t_hat, m_hat])
    RT = R_Ci_N.T
    top = np.hstack([RT,np.zeros((3,3))])
    bot = np.hstack([np.zeros((3,3)),RT])
    return np.vstack([top, bot])

# H matrix for the hard-contact model (no friction, no moment transfer)
def H_matrix():
    H_F = np.eye(3)
    H_M = np.zeros((3,3))
    top = np.hstack([H_F,np.zeros((3,3))])
    bot = np.hstack([np.zeros((3, 3)),H_M])
    return np.vstack([top,bot])

def grasp_matrix(center, contacts):
    G, H_blocks = [], []
    for pos, n in contacts:
        n = np.array(n) / np.linalg.norm(n)
        t = np.array([-n[1], n[0], 0.0])
        m = np.array([0.0, 0.0, 1.0])
        Gi_T = R_matrix(n, t, m) @ P_matrix(center, pos)
        G.append(Gi_T)
        H_blocks.append(H_matrix())
    n_c = len(contacts)
    G_tilde_T = np.vstack(G)
    H = np.zeros((6 * n_c, 6 * n_c))
    for i, Hi in enumerate(H_blocks):
        H[6*i:6*i+6, 6*i:6*i+6] = Hi
    G_T = H @ G_tilde_T
    G_full = G_T.T
    keep_rows = [0, 1, 5]
    keep_cols = []
    for i in range(n_c):
        keep_cols.append(6*i)
        keep_cols.append(6*i+1)
        keep_cols.append(6*i+2)
    Gp = G_full[np.ix_(keep_rows, keep_cols)]
    return G_full, Gp


center = np.array([0.0, 0.0, 0.0])
p4x = 0.0   

contact_1 = (np.array([-1.0,1.5,0.0]),np.array([0.0,-1.0,0.0]))
contact_2 = (np.array([ 1.0,1.5,0.0]),np.array([0.0,-1.0,0.0]))
contact_3 = (np.array([-3.0,0.5,0.0]),np.array([1.0,0.0,0.0]))
contact_4 = (np.array([p4x,-1.5,0.0]),np.array([0.0,1.0,0.0]))
fixed_contacts = [contact_1,contact_2,contact_3,contact_4]

G_full, Gp_test = grasp_matrix(center,fixed_contacts+[(np.array([3.0,0.0,0.0]), np.array([-1.0,0.0,0.0]))])
print("Full G shape:", G_full.shape)
print("Gp shape:", Gp_test.shape)
print("Gp (5 contacts, 5th at (3,0) right edge):\n", np.round(Gp_test, 3))


def boundary_point(s):
    seg_lens = [2*A, 2*B, 2*A, 2*B]
    starts = [(-A, -B), (A, -B), (A, B), (-A, B)]
    dirs   = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    normals = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    s = s % sum(seg_lens)
    for L, st, d, n in zip(seg_lens, starts, dirs, normals):
        if s <= L:
            pos = (st[0] + d[0]*s, st[1] + d[1]*s)
            return np.array([pos[0], pos[1], 0.0]), np.array([n[0], n[1], 0.0])
        s -= L
    raise RuntimeError("unreachable")

perimeter = 4*A + 4*B
step = 0.1
s_vals = np.arange(0, perimeter, step)

def quality_metrics(Gp):
    r = np.linalg.matrix_rank(Gp)
    svals = np.linalg.svd(Gp, compute_uv=False)
    sigma_min, sigma_max = svals[-1], svals[0]
    vol = np.prod(svals)
    if sigma_max > 1e-12:
        iso = sigma_min / sigma_max
    else:
        print("Warning: sigma_max is very small, may indicate rank deficiency.")
        iso = 0.0
    return r, sigma_min, vol, iso

# --- initial random 5th-contact check (assignment step 1) ---
np.random.seed(0)
s_random = np.random.uniform(0, perimeter)
pos_r, n_r = boundary_point(s_random)
_, Gp_random = grasp_matrix(center, fixed_contacts + [(pos_r, n_r)])
r_random, smin_random, _, _ = quality_metrics(Gp_random)
print(f"\nInitial random 5th-contact location: s={s_random:.2f} cm, pos=({pos_r[0]:.2f},{pos_r[1]:.2f})")
print("Gp at this location:\n", np.round(Gp_random, 3))
print(f"rank(Gp) = {r_random} -> {'worth computing quality' if r_random == 3 else 'rank-deficient, skip'}")
print(f"Q_MSV at this location = {smin_random:.4f}")

ranks, q_msv, q_vol, q_iso = [], [], [], []
for s in s_vals:
    pos, n = boundary_point(s)
    _, Gp = grasp_matrix(center, fixed_contacts + [(pos, n)])
    r, smin, vol, iso = quality_metrics(Gp)
    ranks.append(r); q_msv.append(smin); q_vol.append(vol); q_iso.append(iso)

ranks = np.array(ranks); q_msv = np.array(q_msv); q_vol = np.array(q_vol); q_iso = np.array(q_iso)

print("\nAny rank-deficient locations (rank<3)?", np.any(ranks < 3))
print("Min rank observed:", ranks.min())

def report_best(name, q):
    i = np.argmax(q)
    pos, n = boundary_point(s_vals[i])
    print(f"{name}: best s={s_vals[i]:.1f} cm, pos=({pos[0]:.2f},{pos[1]:.2f}), value={q[i]:.4f}")
    return i

i_msv = report_best("Q_MSV (3.1.1)", q_msv)
i_vol = report_best("Q_VOL (3.1.2)", q_vol)
i_iso = report_best("Q_ISO (3.1.3)", q_iso)

# Plotting code
fig2, ax2 = plt.subplots(figsize=(6, 4))
rect_x = [-A, A, A, -A, -A]; rect_y = [-B, -B, B, B, -B]
ax2.plot(rect_x, rect_y, 'b-')
for i, (pos, n) in enumerate(fixed_contacts):
    ax2.arrow(pos[0]-0.6*n[0], pos[1]-0.6*n[1], 0.5*n[0], 0.5*n[1], head_width=0.15, color='red',
               label='fixed contacts' if i == 0 else None)

pos_msv, n_msv = boundary_point(s_vals[i_msv])
ax2.arrow(pos_msv[0]-0.6*n_msv[0], pos_msv[1]-0.6*n_msv[1], 0.5*n_msv[0], 0.5*n_msv[1],
           width=0.05, head_width=0.25, length_includes_head=True, color='green', label='best (MSV)')

pos_vol, n_vol = boundary_point(s_vals[i_vol])
ax2.arrow(pos_vol[0]-0.6*n_vol[0], pos_vol[1]-0.6*n_vol[1], 0.5*n_vol[0], 0.5*n_vol[1],
           head_width=0.15, color='purple', label='best (VOL)')

pos_iso, n_iso = boundary_point(s_vals[i_iso])
ax2.arrow(pos_iso[0]-0.6*n_iso[0], pos_iso[1]-0.6*n_iso[1], 0.5*n_iso[0], 0.5*n_iso[1],
           head_width=0.15, color='orange', label='best (ISO)')

ax2.set_aspect('equal'); ax2.legend(); ax2.set_title('Best 5th-contact location per metric')
plt.savefig('results/best_locations.png', dpi=150)
print("Saved diagram.")