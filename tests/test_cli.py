import subprocess


def test_cli_help_runs():
    cmds = [
        ["train-gtsrb", "--help"],
        ["analyze-results", "--help"],
        ["visualize-filters", "--help"],
        ["visualize-activations", "--help"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()
