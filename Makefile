PYTHON = uv run python
SRC = src/pitpattern

.PHONY: setup test target-figures model-comparison \
	pit-crypt-explore pit-crypt-validate pit-crypt-sweep pit-crypt-figures \
	pit-groove-explore pit-groove-validate \
	goal-figures voronoi cpm particles figueiredo buckling \
	pit-goal-spatial pit-goal-smoke pit-goal-sweep pit-goal-ramp pit-goal-compare pit-goal-robustness \
	pit-ring pit-ring-validate pit-microphase pit-microphase-validate pit-activator-inhibitor \
	sweep-video dbm-phases dbm-video dbm-grid dbm-compare dbm-v2-video dbm-v2-phases dbm-v2-compare dbm-v2-vs-real \
	run exp exp2 exp3 exp4 short

# ---------------------------------------------------------------- environment / tests
setup:
	uv sync --extra dev

test:
	${PYTHON} -m pytest tests/ -v

# ---------------------------------------------------------------- goal and reference figures (docs/images)
target-figures:
	${PYTHON} ${SRC}/figures/make_pit_ideal_sweep.py
	${PYTHON} ${SRC}/figures/make_pit_reference_sheet.py
	${PYTHON} ${SRC}/figures/make_pit_target_schema.py

model-comparison:
	${PYTHON} ${SRC}/figures/make_model_comparison.py

# ---------------------------------------------------------------- crypt lobulation model (docs/models/pit_crypt_model.md)
CRYPT_PS = 0.1 0.2 0.3 0.8 0.9 1.0 2.2 2.3 2.4 3.0 3.2 3.4

pit-crypt-explore:
	${PYTHON} ${SRC}/models/simulate_pit_crypt.py scan --output artifacts/pit_crypt/explore --seeds 1 2 3 --ps 0.3 0.9 1.6 2.3 3.3 --hold 500

pit-crypt-validate:
	for s in $$(seq 100 119); do ${PYTHON} ${SRC}/models/simulate_pit_crypt.py scan --output artifacts/pit_crypt/validation/seed$$s --seeds $$s --ps ${CRYPT_PS} --hold 500 2000; done
	${PYTHON} ${SRC}/evaluation/evaluate_pit_crypt.py 'artifacts/pit_crypt/validation/seed*/metrics.csv' --json artifacts/pit_crypt/validation/summary.json

pit-crypt-sweep:
	${PYTHON} ${SRC}/models/simulate_pit_crypt.py sweep --output artifacts/pit_crypt/sweep --seed 1 --p-left -0.5 --p-right 3.4 --sweeps 2000

pit-crypt-figures:
	${PYTHON} ${SRC}/figures/make_pit_crypt_figures.py

# ---------------------------------------------------------------- crest-groove model (docs/models/pit_groove_model.md)
GROOVE_DRIVES = -0.40 -0.375 -0.35 -0.30 -0.25 -0.20 -0.15 -0.125 -0.10 -0.075 -0.05 -0.025 0.0 0.0125 0.025 0.0375 0.05 0.075 0.10 0.125 0.15 0.20

pit-groove-explore:
	${PYTHON} ${SRC}/models/simulate_pit_groove.py sweep --output artifacts/pit_groove/explore --seeds 1 7 42 --steps 8000 --checkpoints 1 --drives ${GROOVE_DRIVES}

pit-groove-validate:
	for s in $$(seq 100 119); do ${PYTHON} ${SRC}/models/simulate_pit_groove.py sweep --output artifacts/pit_groove/validation/seed$$s --seeds $$s --steps 8000 --checkpoints 1 --drives ${GROOVE_DRIVES}; done
	${PYTHON} ${SRC}/evaluation/evaluate_pit_groove.py 'artifacts/pit_groove/validation/seed*/metrics.csv' --w0 6.13 --json artifacts/pit_groove/validation/summary.json
	${PYTHON} ${SRC}/figures/make_pit_groove_figures.py

# ---------------------------------------------------------------- earlier candidate models
goal-figures:
	${PYTHON} ${SRC}/figures/make_goal_figures.py --output-dir docs/images

voronoi:
	${PYTHON} ${SRC}/models/simulate_pit_voronoi.py

cpm:
	${PYTHON} ${SRC}/models/simulate_pit_cpm.py

particles:
	${PYTHON} ${SRC}/models/simulate_pit_particles.py

figueiredo:
	${PYTHON} ${SRC}/models/simulate_pit_figueiredo.py

buckling:
	${PYTHON} ${SRC}/models/simulate_pit_buckling.py

pit-goal-spatial:
	${PYTHON} ${SRC}/models/simulate_pit_spatial.py --output artifacts/pit_spatial --size 200 --steps 5500 --dt 0.28 --seed 7

pit-goal-smoke:
	${PYTHON} ${SRC}/evaluation/evaluate_pit_goal.py smoke --model sh_inhibitor

pit-goal-sweep:
	${PYTHON} ${SRC}/evaluation/evaluate_pit_goal.py sweep --model sh_inhibitor --output artifacts/pit_goal_sweep --p-count 25

pit-goal-ramp:
	${PYTHON} ${SRC}/evaluation/evaluate_pit_goal.py ramp --model sh_inhibitor --output artifacts/pit_goal_ramp

pit-goal-compare:
	${PYTHON} ${SRC}/evaluation/evaluate_pit_goal.py compare --output artifacts/pit_goal_compare --p-count 15

pit-goal-robustness:
	${PYTHON} ${SRC}/evaluation/evaluate_pit_goal.py robustness --model sh_inhibitor --output artifacts/pit_goal_robust --seeds 7 11 17 --sizes 96 128

pit-ring:
	${PYTHON} ${SRC}/models/simulate_pit_ring.py

pit-ring-validate:
	${PYTHON} ${SRC}/evaluation/evaluate_pit_ring.py
	${PYTHON} ${SRC}/evaluation/summarize_pit_ring.py

pit-microphase:
	${PYTHON} ${SRC}/models/simulate_pit_microphase.py --output artifacts/pit_microphase/fine --heterogeneity .08 --steps 2500 --means -.55 -.35 -.15 -.05 0 .05 .1 .15 .2 .25

pit-microphase-validate: pit-microphase
	${PYTHON} ${SRC}/evaluation/validate_pit_microphase.py
	${PYTHON} ${SRC}/evaluation/summarize_pit_microphase.py

pit-activator-inhibitor:
	${PYTHON} ${SRC}/models/simulate_pit_activator_inhibitor.py

# ---------------------------------------------------------------- Gray-Scott path sweep video (docs/models/gs-sweep-*-report.md)
sweep-video:
	${PYTHON} ${SRC}/models/sweep_video.py --out artifacts/pit-sweep/pit-sweep.mp4 \
		--phases-out artifacts/pit-sweep/pit-sweep-phases.png \
		--metrics-out artifacts/pit-sweep/pit-sweep-metrics.csv

# ---------------------------------------------------------------- Laplacian growth (DBM) pit model (docs/models/dbm-v1-report.md, dbm-v2-report.md)
DBM_OUT = artifacts/pit-dbm
DBM_ARGS = --size 256 --spacing 22 --r0 2 --r-w 1.5 --r-w-end 2.25 --ell 12 --beta 3 --beta-r 2 --beta-r-end 5 \
	--ramp-start 0.4 --eta 0 --eta-end 4 --eta-hold 0.006 --eta-ramp 0.014 --n-active 30 --dorm-s 0.04 --dorm-fade 0.08 \
	--flow-s 0.6 --flow-rate 0.1 --flow-delta 0.15 --flow-r 3 --mass-max 1900 --seed 11
DBM_KNOTS = 0:0,0.08:0.006,0.18:0.007,0.28:0.04,0.38:0.042,0.60:0.40,0.70:0.41,0.90:0.72,1:0.74

dbm-phases:
	${PYTHON} ${SRC}/models/laplacian_pit.py phases --out ${DBM_OUT} --tag pit-dbm-v1 ${DBM_ARGS} \
		--s-phases 0.006,0.04,0.4,0.72 --scale 3

dbm-video:
	${PYTHON} ${SRC}/models/laplacian_pit.py video --out ${DBM_OUT}/pit-dbm-v1.mp4 \
		--phases-out ${DBM_OUT}/pit-dbm-v1-phases.png ${DBM_ARGS} --duration 45 --fps 30 --scale 3 \
		--metric-every 3 --knots ${DBM_KNOTS}

dbm-grid:
	${PYTHON} ${SRC}/models/laplacian_pit.py grid --out ${DBM_OUT} --tag eta-grid --size 192 --spacing 48 --ell 32 \
		--etas 0,1,2,3,4,6 --r-w 1.0 --beta 4 --beta-r 2 --mass-max 1200 --s-list 0.03,0.08,0.15,0.3,0.5,0.7,1.0 --zoom 128

dbm-compare:
	${PYTHON} ${SRC}/models/laplacian_pit.py compare --gs docs/images/pit-sweep-v2-phases.png \
		--dbm ${DBM_OUT}/pit-dbm-v1-phases.png --out ${DBM_OUT}/gs-v2-vs-dbm-v1.png

# DBM v2: ~195 nuclei, two dormancy stages (195 -> 100 tubes -> 30 trees), noise reduction m=4,
# screening length ramp 12 -> 40, eta 6 -> 0.5 in the villous phase, area-preserving island rounding.
DBM2_ARGS = --size 256 --spacing 19.5 --r0 2.5 --r-w 1.5 --r-w-end 2.25 --ell 12 --ell-end 40 --ell-s0 0.05 --ell-s1 0.3 \
	--beta 3 --beta-r 2 --beta-r-end 5 --ramp-start 0.4 --ramp-end 0.7 --eta 0 --eta-end 6 --eta-late 0.5 \
	--eta-hold 0.006 --eta-ramp 0.014 --stages 0.015:100,0.05:30 --dorm-fade 0.05 --noise-m 4 \
	--island-flow-s 0.5 --island-flow-rate 0.5 --island-flow-r 4 --mass-max 1900 --seed 11 --sor-iters 16 --grow-frac 0.5
DBM2_KNOTS = 0:0,0.08:0.007,0.18:0.008,0.27:0.055,0.35:0.058,0.56:0.30,0.66:0.31,0.88:0.75,1:0.77

dbm-v2-video:
	${PYTHON} ${SRC}/models/laplacian_pit.py video --out ${DBM_OUT}/pit-dbm-v2.mp4 \
		--phases-out ${DBM_OUT}/pit-dbm-v2-phases.png ${DBM2_ARGS} --duration 45 --fps 30 --scale 3 \
		--metric-every 3 --knots ${DBM2_KNOTS}

dbm-v2-phases:
	${PYTHON} ${SRC}/models/laplacian_pit.py phases --out ${DBM_OUT} --tag pit-dbm-v2 ${DBM2_ARGS} \
		--s-phases 0.007,0.055,0.30,0.75 --scale 3

dbm-v2-compare:
	${PYTHON} ${SRC}/models/laplacian_pit.py compare --gs ${DBM_OUT}/pit-dbm-v1-phases.png --gs-title "DBM v1" \
		--dbm ${DBM_OUT}/pit-dbm-v2-phases.png --dbm-title "DBM v2" --out ${DBM_OUT}/pit-dbm-v1-vs-v2.png

# stain-like polarity (pits dark) next to the clinical contact sheet (not in the repo: mixed CC BY-NC-ND licences)
PIT_REF ?= media/pit-reference/contact-sheet.png
dbm-v2-vs-real:
	${PYTHON} ${SRC}/models/laplacian_pit.py sheet --run ${DBM_OUT}/pit-dbm-v2.json --polarity pit-dark \
		--out ${DBM_OUT}/pit-dbm-v2-phases-dark.png
	${PYTHON} ${SRC}/models/laplacian_pit.py compare --gs ${PIT_REF} \
		--gs-title "Clinical crystal-violet pit patterns (Kudo I / IIIL / IV branching / IV villous)" \
		--dbm ${DBM_OUT}/pit-dbm-v2-phases-dark.png --dbm-title "DBM v2 (pits rendered dark)" \
		--out ${DBM_OUT}/pit-dbm-v2-vs-real.png

# ---------------------------------------------------------------- legacy interactive Gray-Scott tools (pygame)
run:
	${PYTHON} legacy/pit.py

exp:
	${PYTHON} legacy/exp.py

exp2:
	${PYTHON} legacy/exp2.py

exp3:
	${PYTHON} legacy/exp3.py

exp4:
	${PYTHON} legacy/exp4.py

short:
	${PYTHON} legacy/short.py
