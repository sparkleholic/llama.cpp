import base64
from openai import OpenAI
from PIL import Image

client = OpenAI(base_url="http://localhost:8888/v1", api_key="xxx")
# client = OpenAI(base_url="http://10.177.124.98:8888/v1", api_key="xxx")


# Function to encode the image
def encode_image(image_path, image_data=None):
    if image_path is not None:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
    if image_data is not None:
        return base64.b64encode(image_data).decode("utf-8")
    return None


# Path to your image
image_path = "./mario.jpeg"

# print resolution of the image
image = Image.open(image_path)
width, height = image.size
print(f"Resolution: {width}x{height}")

# Getting the Base64 string
base64_image = encode_image(image_path=image_path)
print("***********")
print(base64_image)
print("***********")

# SYSTEM_PROMPT = """
#     You are a helpful assistant capable of accessing external functions and engaging in casual chat.
#     Use the responses from these function calls to provide accurate and informative answers.
#     The answers should be natural and hide the fact that you are using tools to access real-time information.
#     Guide the user about available tools and their capabilities.
#     Always utilize tools to access real-time information when required.
#     Engage in a friendly manner to enhance the chat experience.
# """

SYSTEM_PROMPT = """
    You are a helpful assistant capable of accessing external functions and engaging in casual chat.
    Engage in a friendly manner to enhance the chat experience.
    Answer in a phrase which start with 'This image is ' and do not make multiple sentences.
"""


# chat_completion = client.chat.completions.create(
#     model="llama-3",  # Model name, adjust as necessary
#     messages=[
#         {
#             "role": "system",
#             "content": SYSTEM_PROMPT
#         },
#         {
#             "role": "user",
#             "content": "What is in the image?"
#         }
#     ],
# )

chat_completion = client.chat.completions.create(
    model="llama-3",  # Model name, adjust as necessary
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in the image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
    max_tokens=128
)

print(chat_completion.choices[0].message.content)
