import argparse
import base64

def encode_image_to_base64(image_path):
    """Read an image file and encode it as base64."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an image file to base64.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file.")
    args = parser.parse_args()

    base64_text = encode_image_to_base64(args.image)
    if base64_text:
        print("Base64 Encoded Text:")
        print(base64_text)
