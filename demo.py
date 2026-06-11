from src.object_recon.gradio_app import build_demo


def main() -> None:
    demo = build_demo()
    try:
        demo.launch(inbrowser=True, share=False)
    finally:
        demo.close()


if __name__ == "__main__":
    main()
