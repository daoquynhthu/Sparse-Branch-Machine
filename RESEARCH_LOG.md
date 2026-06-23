# Research log

## Baseline

Imported v2 unchanged as the first Git commit. The baseline couples stable logical IDs to movable physical slots, uses hot/cold address buckets, and reaches 96.875% frozen evaluation accuracy on the default deterministic FSM dataset.

## Architecture v3

This branch adds architectural semantics without replacing the v2 storage or execution core:

- finite-horizon route traces and decayed node credit;
- edge eligibility state;
- explicit node lifecycle: cold, warm, mature, dormant;
- parent-linked specialization nodes;
- structural split diagnostics;
- conservative redundant-node merge;
- periodic merge control in the benchmark CLI.

### Negative result retained

An early revision treated a wrong final prediction as negative evidence against every traversed control edge. Frozen evaluation accuracy fell from 96.875% to 78.125%. The inference was invalid: outcome failure does not identify which traversed edge was causally wrong. That edge punishment was removed. Current code keeps delayed node-level utility credit but only reinforces traversed edges weakly after errors until a counterfactual routing test exists.

### Current default result

On the same deterministic dataset:

- train accuracy: 99.73%;
- frozen evaluation accuracy: 93.75%;
- 100 live nodes;
- 54 conflict-triggered specialization events;
- about 4 active nodes and 18.4 examined candidates per step.

This is still below the v2 accuracy baseline. The branch is therefore an architectural research branch, not a replacement release. The regression is explicit and should be resolved by improving specialization selection rather than reverting to unconstrained growth or hiding the result.
