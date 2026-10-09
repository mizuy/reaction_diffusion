PYTHON = uv run python
SRC = src/pitpattern

.PHONY: setup test target-figures model-comparison \
	pit-groove-explore pit-groove-validate \
	goal-figures voronoi cpm particles figueiredo buckling \
	pit-goal-spatial pit-goal-smoke pit-goal-sweep pit-goal-ramp pit-goal-compare pit-goal-robustness \
	pit-ring pit-ring-validate pit-microphase pit-microphase-validate pit-activator-inhibitor \
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
