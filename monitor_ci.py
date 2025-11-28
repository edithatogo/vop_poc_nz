import json
import subprocess
import time


def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{cmd}': {e}")
        return None

def get_latest_run_id():
    # Get the latest run ID for the main branch
    cmd = "gh run list --branch main --limit 1 --json databaseId,status,conclusion"
    output = run_command(cmd)
    if output:
        runs = json.loads(output)
        if runs:
            return runs[0]
    return None

def monitor_run(run_id):
    print(f"Monitoring Run ID: {run_id}...")
    while True:
        cmd = f"gh run view {run_id} --json status,conclusion"
        output = run_command(cmd)
        if output:
            status_data = json.loads(output)
            status = status_data.get("status")
            conclusion = status_data.get("conclusion")

            print(f"Current Status: {status}, Conclusion: {conclusion}")

            if status == "completed":
                return conclusion

        time.sleep(10)

def get_failed_logs(run_id):
    print(f"Fetching logs for failed run {run_id}...")
    # Fetch the log for the failed step.
    # We look for the 'Test with pytest' step failure.
    cmd = f"gh run view {run_id} --log-failed"
    output = run_command(cmd)
    if not output:
        print("No failed logs found via --log-failed. Fetching full log...")
        cmd = f"gh run view {run_id} --log"
        output = run_command(cmd)
    return output

def main():
    print("Starting CI Monitor...")

    # 1. Find latest run
    latest_run = get_latest_run_id()
    if not latest_run:
        print("No runs found.")
        return

    run_id = latest_run['databaseId']
    print(f"Found latest run: {run_id} (Status: {latest_run['status']})")

    # 2. Monitor until complete
    conclusion = monitor_run(run_id)

    if conclusion == "success":
        print("Run passed successfully! ✅")
    else:
        print(f"Run failed with conclusion: {conclusion} ❌")

        # 3. Get logs and extract DEBUG info
        logs = get_failed_logs(run_id)
        if logs:
            print("\n--- DEBUG OUTPUT FROM LOGS ---")
            # Filter for our DEBUG prints
            debug_lines = [line for line in logs.splitlines() if "DEBUG:" in line]
            for line in debug_lines:
                print(line)

            print("\n--- FAILURE DETAILS ---")
            # Print the assertion error part
            lines = logs.splitlines()
            for i, line in enumerate(lines):
                if "AssertionError" in line or "FAILED" in line:
                    print(line)
                    # Print context
                    for j in range(1, 5):
                        if i+j < len(lines):
                            print(lines[i+j])

if __name__ == "__main__":
    main()
