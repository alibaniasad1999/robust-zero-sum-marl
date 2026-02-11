# DetectorStatus.msg — Transformer disturbance detector output
# Published at control frequency for monitoring and debugging

builtin_interfaces/Time stamp

# Disturbance probability from transformer (0=nominal, 1=disturbance detected)
float32 disturbance_probability

# Blending weight alpha (1=use pi_opt fully, 0=use pi_rob fully)
float32 blending_weight

# Inference latency of the full control loop iteration (ms)
float32 latency_ms

# Raw action arrays
float32[] action_optimal
float32[] action_robust
float32[] action_blended

# Dimensions (for runtime validation)
uint32 obs_dim
uint32 act_dim
uint32 sequence_length
