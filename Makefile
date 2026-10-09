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

voronoi:
	${PYTHON} src/simulate_pit_voronoi.py

cpm:
	${PYTHON} src/simulate_pit_cpm.py

particles:
	${PYTHON} src/simulate_pit_particles.py

figueiredo:
	${PYTHON} src/simulate_pit_figueiredo.py

buckling:
	${PYTHON} src/simulate_pit_buckling.py

goal-figures:
	${PYTHON} src/make_goal_figures.py --output-dir docs/images

pit-goal-spatial:
	${PYTHON} src/simulate_pit_spatial.py --output artifacts/pit_spatial --size 200 --steps 5500 --dt 0.28 --seed 7

pit-goal-smoke:
	${PYTHON} src/evaluate_pit_goal.py smoke --model sh_inhibitor

pit-goal-sweep:
	${PYTHON} src/evaluate_pit_goal.py sweep --model sh_inhibitor --output artifacts/pit_goal_sweep --p-count 25

pit-goal-ramp:
	${PYTHON} src/evaluate_pit_goal.py ramp --model sh_inhibitor --output artifacts/pit_goal_ramp

pit-goal-compare:
	${PYTHON} src/evaluate_pit_goal.py compare --output artifacts/pit_goal_compare --p-count 15

pit-goal-robustness:
	${PYTHON} src/evaluate_pit_goal.py robustness --model sh_inhibitor --output artifacts/pit_goal_robust --seeds 7 11 17 --sizes 96 128

test:
	${PYTHON} -m pytest tests/ -v

pit-ring:
	${PYTHON} src/simulate_pit_ring.py

pit-ring-validate:
	${PYTHON} src/evaluate_pit_ring.py
	${PYTHON} src/summarize_pit_ring.py

pit-microphase:
	${PYTHON} src/simulate_pit_microphase.py --output artifacts/pit_microphase/fine --heterogeneity .08 --steps 2500 --means -.55 -.35 -.15 -.05 0 .05 .1 .15 .2 .25

pit-microphase-validate: pit-microphase
	${PYTHON} src/validate_pit_microphase.py
	${PYTHON} src/summarize_pit_microphase.py

pit-activator-inhibitor:
	${PYTHON} src/simulate_pit_activator_inhibitor.py
