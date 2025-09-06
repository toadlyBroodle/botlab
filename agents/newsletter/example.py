import argparse

from .agents import NewsletterPipeline


def main(args: argparse.Namespace) -> None:
    pipeline = NewsletterPipeline()
    final_md = pipeline.run(topic=args.topic, audience=args.audience, style=args.style, tweet=args.tweet)
    print("\n===== NEWSLETTER (FINAL) =====\n")
    print(final_md)


if __name__ == "__main__":
    """Main function to parse arguments."""
    parser = argparse.ArgumentParser(description="Generate a weekly newsletter from a topic using botlab agents")
    parser.add_argument("--topic", type=str, required=True, help="Topic or focus for this week's issue")
    parser.add_argument("--audience", type=str, help="Target audience (e.g., product managers)")
    parser.add_argument("--style", type=str, help="Style hint (e.g., concise, punchy)")
    parser.add_argument("--tweet", action="store_true", help="Also generate a 280-char tweet summary")
    main(parser.parse_args())


