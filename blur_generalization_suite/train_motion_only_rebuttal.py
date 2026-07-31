import sys

from blur_generalization_suite.train_paper_backbones import main as paper_training_main


def main() -> None:
    if "--training-profile" not in sys.argv:
        sys.argv.extend(["--training-profile", "strict_motion"])
    paper_training_main()


if __name__ == "__main__":
    main()
