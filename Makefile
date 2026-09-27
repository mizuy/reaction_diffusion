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
