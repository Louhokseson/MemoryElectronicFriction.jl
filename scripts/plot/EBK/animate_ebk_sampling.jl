# =============================================================================
# EBK initial-vibration sampling, animated — Gardner–Habershon–Maurer NO/Au(111)
# (the 2-D POGO model used in run_md.jl).
#
# A self-reminder of the two-step logic behind `QuantisedDiatomic.generate_1D_
# vibrations` (run_md.jl) and the quantisation in run_vib_state_noau.jl:
#
#   STEP 1 (quantise / "fit"):  the 1-D NO bond binding curve V(r) is the POGO
#       ground adiabatic surface frozen far from the surface (z = 10 Å). An
#       INTEGER vibrational quantum number ν is mapped to a bond energy Eν by
#       the EBK rule  ∮ p dr = 2π(ν+½)ℏ   ⇔   √(2μ)/π ∫√(Eν−V) dr = ν+½.
#
#   STEP 2 (sample):  at fixed Eν the bond is a 1-D oscillator between turning
#       points r∓. We draw snapshots (r, ṙ) uniformly in TIME along that orbit;
#       ½μṙ² = Eν − V(r) fixes the speed. Because the bond is slow near r∓ it
#       lingers there → r piles up at the turning points; the kinetic energy
#       piles up at its maximum  Eν − V_min  (= "E_above_well").
#
# The POGO binding curve is re-implemented here in atomic units (verified to
# match build_V_eff(): V_min=-0.1035 eV, r_eq=1.152 Å, E₁₆=3.238 eV) so the
# script stays light (CairoMakie only — no NQCDynamics load).
#
# Output: plots/ebk_sampling/ebk_sampling_nu16.gif
# =============================================================================

using DrWatson
@quickactivate "HokseonReproduce"
using CairoMakie
using Printf
CairoMakie.activate!()

# ---------------------------------------------------------------- units (a.u.)
const Ha2eV  = 27.211386245988
const Bohr2Å = 0.52917721067
const Å2Bohr = 1 / Bohr2Å
const u2me   = 1822.888486209
const at2fs  = 0.0241888432651
const vfac   = Bohr2Å / at2fs          # a.u. velocity → Å/fs

invÅ2au(x) = x * Bohr2Å
Å2au(x)    = x * Å2Bohr
eV2au(x)   = x / Ha2eV

# --------------------------------------------------- POGO 1-D bond binding curve
const P = [1.9535329365277612, -0.26876384605775233, 6.571336959902485,
           2.519431394450958,   1.295009818269084,   4.152792292961294,
           1.0015030848580035,  1.2350312403781256,  2.417096730391661,
           8.958749759728342]
morse(a, x0, D, x) = D * (exp(-2a*(x-x0)) - 2exp(-a*(x-x0)))
morseNO(r) = morse(invÅ2au(2.7968), Å2au(1.1510), eV2au(6.610), r)
U0(r, z) = morseNO(r) + exp(-invÅ2au(P[1])*(z - Å2au(P[2]))) + eV2au(P[3])
U1(r, z) = morse(invÅ2au(P[4]), Å2au(P[5]), eV2au(P[6]), r) +
           morse(invÅ2au(P[7]), Å2au(P[8]), eV2au(P[9]), z) + eV2au(P[10])
const V̄k = sqrt(eV2au(1.5) / (2π))
Vk(z) = V̄k * (1 - tanh(z / Å2au(10.0)))
function Vbond(r; z = Å2au(10.0))
    a = U0(r, z); b = U1(r, z); c = Vk(z)
    return 0.5 * (a + b - sqrt((a - b)^2 + 4c^2))
end
const μ = (14.007 * 15.999 / (14.007 + 15.999)) * u2me

# --------------------------------------------------------------- EBK machinery
const RG = collect(Å2au(0.6):Å2au(0.002):Å2au(4.0))
const VG = Vbond.(RG)
function nu_of_E(E)
    f = @. sqrt(max(E - VG, 0.0))
    integral = sum((f[1:end-1] .+ f[2:end]) ./ 2 .* diff(RG))
    return sqrt(2μ) / π * integral - 0.5
end
function E_for_nu(target; lo, hi)
    for _ in 1:200
        m = (lo + hi) / 2
        nu_of_E(m) < target ? (lo = m) : (hi = m)
    end
    return (lo + hi) / 2
end
allowed_idx(E) = findall(<=(E), VG)
function turning_points(E)
    idx = allowed_idx(E); return (RG[first(idx)], RG[last(idx)])
end
# phase-space loop p=±√(2μ(E−V)) → (r[Å], ṙ[Å/fs]); closed
function loop_pts(E)
    idx = allowed_idx(E); rs = RG[idx]
    vs  = @. sqrt(max(2*(E - VG[idx])/μ, 0.0)) * vfac
    up  = Point2f.(rs .* Bohr2Å,  vs)
    dn  = Point2f.(reverse(rs) .* Bohr2Å, reverse(-vs))
    return vcat(up, dn, up[1:1])
end

const Vmin, imin = findmin(VG)
const r_eq  = RG[imin]
const Vout  = VG[end]
const Eladder = [E_for_nu(float(ν); lo = Vmin + 1e-9, hi = Vout - 1e-9) for ν in 0:16]
const E16   = Eladder[end]
const r1au, r2au = turning_points(E16)

# display scalars (eV, Å)
const Vmin_eV = Vmin * Ha2eV
const E16_eV  = E16 * Ha2eV
const KEmax_eV = (E16 - Vmin) * Ha2eV
const r1Å, r2Å, reqÅ = r1au*Bohr2Å, r2au*Bohr2Å, r_eq*Bohr2Å
const vmaxÅfs = sqrt(2*(E16 - Vmin)/μ) * vfac

# ------------------------------------------- classical microcanonical trajectory
dVdr(r) = (Vbond(r + 1e-4) - Vbond(r - 1e-4)) / 2e-4
function trajectory(E; dt = 0.5, nsteps = 12000)
    r = r_eq; v = sqrt(2*(E - Vmin)/μ)        # start at r_eq moving outward
    rs = Vector{Float64}(undef, nsteps); vs = similar(rs)
    for i in 1:nsteps
        rs[i] = r; vs[i] = v
        v += 0.5*dt*(-dVdr(r)/μ); r += dt*v; v += 0.5*dt*(-dVdr(r)/μ)
    end
    return rs, vs
end
const TR, TV = trajectory(E16)
const TKE = @. 0.5 * μ * TV^2 * Ha2eV          # bond kinetic energy, eV
# vibrational period (in steps) from successive bond-length maxima
function period_steps(rs)
    mx = [i for i in 2:length(rs)-1 if rs[i] > rs[i-1] && rs[i] >= rs[i+1]]
    return length(mx) >= 2 ? mx[2]-mx[1] : length(rs)÷4
end
const PSTEP = period_steps(TR)

# --------------------------------------------------------------- histogram bins
hcounts(vals, edges) = [count(e -> edges[k] <= e < edges[k+1], vals) for k in 1:length(edges)-1]
const r_edges  = collect(range(r1Å-0.05, r2Å+0.05, length = 46))
const ke_edges = collect(range(0.0, KEmax_eV*1.04, length = 46))
const r_cent   = (r_edges[1:end-1] .+ r_edges[2:end]) ./ 2
const ke_cent  = (ke_edges[1:end-1] .+ ke_edges[2:end]) ./ 2
# limiting distributions (from the FULL trajectory) → fixes y-limits + target curve
const r_final  = hcounts(TR .* Bohr2Å, r_edges)
const ke_final = hcounts(TKE,          ke_edges)
const r_ymax   = maximum(r_final) * 1.18
const ke_ymax  = maximum(ke_final) * 1.18
# analytic classical density ρ(r) ∝ 1/|v(r)|, scaled to the final histogram
ρr = [Vbond(Å2au(rc)) < E16 ? 1/sqrt(max(2*(E16-Vbond(Å2au(rc)))/μ, 1e-12)) : 0.0 for rc in r_cent]
const ρr_scaled = ρr ./ maximum(ρr) .* maximum(r_final)

# =============================================================================
# Figure + observables
# =============================================================================
col = (well=:black, level=RGBf(0.20,0.30,0.75), hot=RGBf(0.85,0.25,0.20),
       ball=RGBf(0.90,0.45,0.10), samp=RGBf(0.20,0.55,0.35), faint=RGBf(0.6,0.6,0.7))

const MP = projectdir("fonts", "MinionPro-Capt.otf")
fig = Figure(size = (1180, 760), figure_padding = 12, fonts = (; regular = MP, bold = MP))
Label(fig[0, 1:3], "EBK initialisation of NO vibration — Gardner–Habershon–Maurer NO/Au(111) model";
      fontsize = 23, font = :bold, color = RGBf(0.12,0.12,0.18))

o_read = Observable("")   # live read-out shown as the potential-panel title (kept out of the data area)
# shared larger label / tick fonts for clearer presentation
const AXKW = (; xlabelsize = 21, ylabelsize = 21, xticklabelsize = 17, yticklabelsize = 17)
ax_pot = Axis(fig[1:2, 1]; xlabel = "N–O bond length  r / Å", ylabel = "energy / eV",
              title = o_read, titlesize = 19, titlecolor = RGBf(0.20,0.30,0.75), AXKW...)
ax_ph  = Axis(fig[1:2, 2]; xlabel = "r / Å", ylabel = "ṙ / (Å fs⁻¹)",
              title = "phase space  (r, ṙ)", titlesize = 18, AXKW...)
ax_hr  = Axis(fig[1, 3]; xlabel = "sampled r / Å", ylabel = "count",
              title = "STEP 2a: positions", titlesize = 18, AXKW...)
ax_ke  = Axis(fig[2, 3]; xlabel = "sampled KE = ½μṙ² / eV", ylabel = "count",
              title = "STEP 2b: kinetic energy", titlesize = 18, AXKW...)
cap = Observable(" ")
Label(fig[3, 1:3], cap; fontsize = 16, justification = :left, lineheight = 1.1)
colsize!(fig.layout, 3, Relative(0.28))

# static potential curve
lines!(ax_pot, RG .* Bohr2Å, VG .* Ha2eV; color = col.well, linewidth = 2.5)
xlims!(ax_pot, r1Å-0.13, r2Å+0.22); ylims!(ax_pot, Vmin_eV-0.25, E16_eV+0.45)
xlims!(ax_ph,  r1Å-0.13, r2Å+0.22); ylims!(ax_ph, -vmaxÅfs*1.18, vmaxÅfs*1.18)
xlims!(ax_hr, r_edges[1], r_edges[end]); ylims!(ax_hr, 0, r_ymax)
xlims!(ax_ke, ke_edges[1], ke_edges[end]); ylims!(ax_ke, 0, ke_ymax)

# dynamic observables
o_ladder = Observable(Point2f[])                 # revealed integer levels
o_elevel = Observable(Point2f[Point2f(r1Å,Vmin_eV), Point2f(r2Å,Vmin_eV)])
o_tp     = Observable(Point2f[])                 # turning-point verticals
o_ball   = Observable(Point2f[])                 # bead in the well
o_loop   = Observable(loop_pts(Eladder[1]))      # phase-space orbit
o_pmark  = Observable(Point2f[])                 # moving sample point in phase space
o_psamp  = Observable(Point2f[])                 # accumulated phase-space samples
o_hr     = Observable(zeros(length(r_cent)))
o_ke     = Observable(zeros(length(ke_cent)))

linesegments!(ax_pot, o_ladder; color = col.faint, linewidth = 1.2)
lines!(ax_pot, o_elevel; color = col.level, linewidth = 3)
linesegments!(ax_pot, o_tp; color = col.hot, linewidth = 1.6, linestyle = :dash)
scatter!(ax_pot, o_ball; color = col.ball, markersize = 20, strokecolor=:black, strokewidth=1)

lines!(ax_ph, o_loop; color = col.level, linewidth = 2.5)
scatter!(ax_ph, o_psamp; color = (col.samp,0.7), markersize = 6)
scatter!(ax_ph, o_pmark; color = col.ball, markersize = 18, strokecolor=:black, strokewidth=1)
text!(ax_ph, r1Å-0.10, vmaxÅfs*1.05; text = "area ∮p dr = 2π(ν+½)ℏ",
      align = (:left,:top), fontsize = 14, color = col.level)

barplot!(ax_hr, r_cent, o_hr; color = (col.samp,0.85), gap = 0.05)
lines!(ax_hr, r_cent, ρr_scaled; color = col.hot, linewidth = 2, linestyle = :dash)
vlines!(ax_hr, [r1Å, r2Å]; color = col.hot, linewidth = 1, linestyle = :dot)

barplot!(ax_ke, ke_cent, o_ke; color = (col.ball,0.85), gap = 0.05)
vlines!(ax_ke, [KEmax_eV]; color = col.hot, linewidth = 2, linestyle = :dash)
text!(ax_ke, ke_edges[1] + 0.04, ke_ymax*0.97;
      text = @sprintf("max KE = Eν − V_min\n= %.2f eV", KEmax_eV),
      align = (:left,:top), fontsize = 13, color = col.hot)

# =============================================================================
# Storyboard
# =============================================================================
const N1 = 120   # quantise (build ladder) — slowed so newcomers can follow the ladder forming
const N2 = 22    # hold ν = 16  (time to read the locked-in level)
const N3 = 110   # sample  (pace already felt right)
const NF = N1 + N2 + N3

ladder_segs(nrev) = vcat(([Point2f(turning_points(Eladder[k+1])[1]*Bohr2Å, Eladder[k+1]*Ha2eV),
                           Point2f(turning_points(Eladder[k+1])[2]*Bohr2Å, Eladder[k+1]*Ha2eV)]
                          for k in 0:nrev)...)
tp_segs(E) = (r1=turning_points(E)[1]*Bohr2Å; r2=turning_points(E)[2]*Bohr2Å; EeV=E*Ha2eV;
              Point2f[Point2f(r1,Vmin_eV),Point2f(r1,EeV), Point2f(r2,Vmin_eV),Point2f(r2,EeV)])

const CAP1 = "STEP 1 — QUANTISE.  Raise the energy until the EBK action ∮p dr = 2π(ν+½)ℏ equals an\ninteger ν.  Each integer ν fixes one bound bond energy Eν (the levels stacking up at left)."
const CAP2 = @sprintf("ν = 16 LOCKED.  Eν = %.2f eV  (%.2f eV above the well bottom).  Turning points r∓ (red dashed)\nbound the oscillation — the bond is now sampled at this fixed energy.", E16_eV, KEmax_eV)
const CAP3 = "STEP 2 — SAMPLE.  Roll the classical orbit at Eν and record (r, ṙ) at uniform time steps\n(½μṙ² = Eν − V(r)).  Slow near r∓ → r piles at the turning points;  KE = ½μṙ² is U-shaped (mostly ≈ 0, max = Eν − V_min)."

gifpath = projectdir("docs","ebk_sampling_nu16.gif")
mkpath(dirname(gifpath))
@info "Rendering $(NF) frames → $(gifpath)"

function update!(f)
    if f <= N1                                   # ---- ACT 1: quantise
        cap[] = CAP1
        νt = (f-1)/(N1-1) * 16
        E  = E_for_nu(max(νt,0.02); lo = Vmin+1e-9, hi = Vout-1e-9)
        nrev = clamp(floor(Int, nu_of_E(E)+1e-6), 0, 16)
        o_ladder[] = ladder_segs(nrev)
        EeV = E*Ha2eV; r1,r2 = turning_points(E) .* Bohr2Å
        o_elevel[] = Point2f[Point2f(r1,EeV), Point2f(r2,EeV)]
        o_tp[]     = tp_segs(E)
        o_loop[]   = loop_pts(E)
        o_ball[]   = Point2f[]; o_pmark[] = Point2f[]; o_psamp[] = Point2f[]
        o_hr[] = zeros(length(r_cent)); o_ke[] = zeros(length(ke_cent))
        o_read[] = @sprintf("STEP 1 · quantising:   ν(E) = %.2f", nu_of_E(E))
    elseif f <= N1 + N2                          # ---- ACT 2: hold ν = 16
        cap[] = CAP2
        o_ladder[] = ladder_segs(16)
        o_elevel[] = Point2f[Point2f(r1Å,E16_eV), Point2f(r2Å,E16_eV)]
        o_tp[]     = tp_segs(E16)
        o_loop[]   = loop_pts(E16)
        o_ball[]   = Point2f[Point2f(reqÅ, E16_eV)]
        o_pmark[]  = Point2f[Point2f(reqÅ, vmaxÅfs)]
        o_read[]   = @sprintf("ν = 16 locked   ·   Eν = %.2f eV", E16_eV)
    else                                         # ---- ACT 3: sample
        cap[] = CAP3
        k = f - N1 - N2
        span = min(length(TR), 3*PSTEP)
        m = clamp(round(Int, k/N3 * span), 2, length(TR))
        idx = m
        o_ball[]  = Point2f[Point2f(TR[idx]*Bohr2Å, E16_eV)]
        o_pmark[] = Point2f[Point2f(TR[idx]*Bohr2Å, TV[idx]*vfac)]
        st = max(1, m ÷ 350)
        o_psamp[] = Point2f.(TR[1:st:m] .* Bohr2Å, TV[1:st:m] .* vfac)
        o_hr[] = hcounts(TR[1:m] .* Bohr2Å, r_edges)
        o_ke[] = hcounts(TKE[1:m],          ke_edges)
        o_read[] = @sprintf("STEP 2 · sampling ν = 16   ·   %d snapshots", m)
    end
end

record(fig, gifpath, 1:NF; framerate = 20) do f
    update!(f)
end
@info "Saved" gifpath

# optional still frames (set EBK_STILLDIR to emit PNGs; no ffmpeg needed)
if haskey(ENV, "EBK_STILLDIR")
    for (n, fr) in [("act1", 60), ("act2", N1 + 10), ("act3", NF)]
        update!(fr)
        save(joinpath(ENV["EBK_STILLDIR"], "ebk_still_$(n).png"), fig)
    end
end
println("GIF written to: ", gifpath)
