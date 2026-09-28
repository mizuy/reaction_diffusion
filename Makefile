PYTHON = uv run python

run:
	${PYTHON} src/pit.py

exp:
	${PYTHON} src/exp.py

exp2:
	${PYTHON} src/exp2.py

exp3:
	${PYTHON} src/exp3.py

exp4:
	${PYTHON} src/exp4.py

short:
	${PYTHON} src/short.py

sweep-video:
	${PYTHON} src/sweep_video.py --out artifacts/pit-sweep/pit-sweep.mp4 \
		--phases-out artifacts/pit-sweep/pit-sweep-phases.png \
		--metrics-out artifacts/pit-sweep/pit-sweep-metrics.csv

# Laplacian growth (DBM) pit model: src/laplacian_pit.py
DBM_OUT = artifacts/pit-dbm
DBM_ARGS = --size 256 --spacing 22 --r0 2 --r-w 1.5 --r-w-end 2.25 --ell 12 --beta 3 --beta-r 2 --beta-r-end 5 \
	--ramp-start 0.4 --eta 0 --eta-end 4 --eta-hold 0.006 --eta-ramp 0.014 --n-active 30 --dorm-s 0.04 --dorm-fade 0.08 \
	--flow-s 0.6 --flow-rate 0.1 --flow-delta 0.15 --flow-r 3 --mass-max 1900 --seed 11
DBM_KNOTS = 0:0,0.08:0.006,0.18:0.007,0.28:0.04,0.38:0.042,0.60:0.40,0.70:0.41,0.90:0.72,1:0.74

dbm-phases:
	${PYTHON} src/laplacian_pit.py phases --out ${DBM_OUT} --tag pit-dbm-v1 ${DBM_ARGS} \
		--s-phases 0.006,0.04,0.4,0.72 --scale 3

dbm-video:
	${PYTHON} src/laplacian_pit.py video --out ${DBM_OUT}/pit-dbm-v1.mp4 \
		--phases-out ${DBM_OUT}/pit-dbm-v1-phases.png ${DBM_ARGS} --duration 45 --fps 30 --scale 3 \
		--metric-every 3 --knots ${DBM_KNOTS}

dbm-grid:
	${PYTHON} src/laplacian_pit.py grid --out ${DBM_OUT} --tag eta-grid --size 192 --spacing 48 --ell 32 \
		--etas 0,1,2,3,4,6 --r-w 1.0 --beta 4 --beta-r 2 --mass-max 1200 --s-list 0.03,0.08,0.15,0.3,0.5,0.7,1.0 --zoom 128

dbm-compare:
	${PYTHON} src/laplacian_pit.py compare --gs docs/images/pit-sweep-v2-phases.png \
		--dbm ${DBM_OUT}/pit-dbm-v1-phases.png --out ${DBM_OUT}/gs-v2-vs-dbm-v1.png

# DBM v2: ~195 nuclei, two dormancy stages (195 -> 100 tubes -> 30 trees), noise reduction m=4,
# screening length ramp 12 -> 40, eta 6 -> 0.5 in the villous phase, area-preserving island rounding.
DBM2_ARGS = --size 256 --spacing 19.5 --r0 2.5 --r-w 1.5 --r-w-end 2.25 --ell 12 --ell-end 40 --ell-s0 0.05 --ell-s1 0.3 \
	--beta 3 --beta-r 2 --beta-r-end 5 --ramp-start 0.4 --ramp-end 0.7 --eta 0 --eta-end 6 --eta-late 0.5 \
	--eta-hold 0.006 --eta-ramp 0.014 --stages 0.015:100,0.05:30 --dorm-fade 0.05 --noise-m 4 \
	--island-flow-s 0.5 --island-flow-rate 0.5 --island-flow-r 4 --mass-max 1900 --seed 11 --sor-iters 16 --grow-frac 0.5
DBM2_KNOTS = 0:0,0.08:0.007,0.18:0.008,0.27:0.055,0.35:0.058,0.56:0.30,0.66:0.31,0.88:0.75,1:0.77

dbm-v2-video:
	${PYTHON} src/laplacian_pit.py video --out ${DBM_OUT}/pit-dbm-v2.mp4 \
		--phases-out ${DBM_OUT}/pit-dbm-v2-phases.png ${DBM2_ARGS} --duration 45 --fps 30 --scale 3 \
		--metric-every 3 --knots ${DBM2_KNOTS}

dbm-v2-phases:
	${PYTHON} src/laplacian_pit.py phases --out ${DBM_OUT} --tag pit-dbm-v2 ${DBM2_ARGS} \
		--s-phases 0.007,0.055,0.30,0.75 --scale 3

dbm-v2-compare:
	${PYTHON} src/laplacian_pit.py compare --gs ${DBM_OUT}/pit-dbm-v1-phases.png --gs-title "DBM v1" \
		--dbm ${DBM_OUT}/pit-dbm-v2-phases.png --dbm-title "DBM v2" --out ${DBM_OUT}/pit-dbm-v1-vs-v2.png

# stain-like polarity (pits dark) next to the clinical contact sheet (not in the repo: mixed CC BY-NC-ND licences)
PIT_REF ?= media/pit-reference/contact-sheet.png
dbm-v2-vs-real:
	${PYTHON} src/laplacian_pit.py sheet --run ${DBM_OUT}/pit-dbm-v2.json --polarity pit-dark \
		--out ${DBM_OUT}/pit-dbm-v2-phases-dark.png
	${PYTHON} src/laplacian_pit.py compare --gs ${PIT_REF} \
		--gs-title "Clinical crystal-violet pit patterns (Kudo I / IIIL / IV branching / IV villous)" \
		--dbm ${DBM_OUT}/pit-dbm-v2-phases-dark.png --dbm-title "DBM v2 (pits rendered dark)" \
		--out ${DBM_OUT}/pit-dbm-v2-vs-real.png
