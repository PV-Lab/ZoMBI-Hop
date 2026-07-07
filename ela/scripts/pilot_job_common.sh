# Back-compat shim — canonical file is slurm/pilot_job_common.sh
# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/slurm/pilot_job_common.sh"
