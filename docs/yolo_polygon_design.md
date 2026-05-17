# YOLO PolygonPlus for PS2.0 Parking Slots

This project should not simply copy HPS-Net. The proposed design keeps the
YOLO-style speed advantage, but adds PS2.0-specific supervision that is useful
for parking-line drawing:

1. Ordered quadrilateral prediction:
   - entrance_left
   - entrance_right
   - rear_right
   - rear_left
2. Entrance semantics:
   - the first two points are always the slot entrance line
   - the model can be evaluated by the original PS2.0 entrance-junction metric
3. Direction consistency:
   - an extra body direction target is stored for each slot
   - predicted rear points should agree with the entrance-to-rear direction
4. Edge-aware refinement:
   - generated labels keep enough information for later polygon edge alignment
   - inference can optionally refine rear depth/angle using image edges

## Label Format

The converter writes two aligned files per image.

Standard YOLO label (`.txt`):

```text
class cx cy w h
```

PolygonPlus extra label (`.poly`):

```text
x1 y1 x2 y2 x3 y3 x4 y4 body_dx body_dy slot_type
```

All coordinates are normalized to `[0, 1]`.

- `cx cy w h` is the axis-aligned box enclosing the quadrilateral, kept in
  normal YOLO format so the standard dataloader can still assign anchors.
- `(x1, y1)` and `(x2, y2)` are ordered entrance endpoints.
- `(x3, y3)` and `(x4, y4)` are rear endpoints.
- `(body_dx, body_dy)` is a unit vector from entrance to rear. In the current
  PS2.0 conversion, this uses the same direction as the marking-point direction
  stored in the original JSON.
- `slot_type` is:
  - `0`: perpendicular-like
  - `1`: parallel/long-entrance-like
  - `2`: angled/parallelogram-like

## Why This Is Different From Plain HPS-Net

HPS-Net uses a polygon representation and polygon-corner loss. PolygonPlus adds
explicit entrance ordering, body direction supervision, type-aware depth priors,
and a planned edge-alignment term. These targets are derived from PS2.0's
directional marking points and `slots` pair annotations, so the model is trained
on the failure case that the current GNN missed: long entrance-line pairs.

## Implementation Phases

1. Generate reliable PolygonPlus labels from PS2.0 JSON.
2. Train a YOLO-like detector with extra polygon regression channels.
3. Add losses:
   - normal YOLO box/class losses
   - ordered corner SmoothL1 loss
   - entrance-line angle/length loss
   - body-direction cosine loss
   - optional edge-alignment loss
4. Add inference and visualization that draws the whole quadrilateral directly.
