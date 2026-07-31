# Deployment (sketch, not deployed)

`infra/aws/` is a Terraform sketch showing how this repo would deploy to
AWS — demonstration of IaC competence, never `terraform apply`'d against a
real account.

## What it provisions

One GPU instance (`g5.2xlarge` by default — 1x A10G, 24GB VRAM, 8 vCPU) running
the entire Ray cluster: `LearnerActor`, `DataActor`s, `ReplayBufferActor`,
`ReanalyzeActor`, all in one process on one box, exactly like `python train.py`
does locally today. Boots off an AWS Deep Learning AMI (NVIDIA driver + Docker
+ NVIDIA Container Toolkit preinstalled), clones the repo, builds the
[Dockerfile](../Dockerfile) from Phase 3, and runs it.

## Why one instance, not a GPU node + CPU node

The obvious next step for a Ray-based system is splitting actors across
nodes — a GPU node for the learner, cheaper CPU nodes for the data actors.
That doesn't work as-is here: `DataActor` and `ReanalyzeActor` both call
`model.recurrent_inference` during MCTS, which runs on whatever GPU JAX finds
in-process — confirmed in profiling logs (`Using GPU: cuda:0` printed by
every actor type, not just the learner). A CPU-only worker node would have no
GPU for those actors to use, and Ray's scheduler has no way to know that. Real
multi-node support needs a code change first — a CPU JAX backend fallback path
for `DataActor` — which is out of scope for a deployment sketch.

## Usage (illustrative — requires an AWS account and a key pair)

```bash
cd infra/aws
terraform init
terraform plan -var="key_name=your-ec2-key" -var="ssh_cidr=$(curl -s ifconfig.me)/32"
# terraform apply  — never run against a real account for this project
```

`wandb_api_key` is optional; omit it and the container runs with
`train.wandb_mode=disabled`.

## Future extensions (not built)

- **Ray autoscaler YAML** — Ray has its own cluster-launcher config
  (`ray up cluster.yaml`) that handles multi-node bring-up and autoscaling
  natively. Skipped here to keep the sketch to one file; would replace the
  single-instance `aws_instance` resource with a head node + worker node
  group once the CPU-fallback code change above lands.
- **Multi-node split** — once `DataActor` can run MCTS inference on CPU-only
  JAX, split into a GPU instance (learner + reanalyze) and N cheaper CPU
  instances (data actors), matching the resource layout description in
  `CLAUDE.md` but at cloud scale.
- **Spot instances** — training is checkpoint-resumable
  (`checkpoint_interval`), so a spot-backed worker pool would be a cheap,
  low-risk win once this is real infrastructure rather than a sketch.
