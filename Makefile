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
