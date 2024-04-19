## Features
Image Text Overlay: Add customizable text overlays to images.
Prompt Selection and Scheduling: Manage and format string prompts based on configurable parameters.
Loop Constructs: Implement looping mechanisms which can reset based on conditions.
Special Comparators: Utilize a unique string class for special comparison features.

## Installation
To run the scripts, you need Python 3.6 or higher. Dependencies are listed in requirements.txt, which can be installed using:

pip install -r requirements.txt

## Requirements

Pillow for image processing (PIL)
PyTorch (torch) for tensor operations
Numpy (numpy)

## Methods
# add_text_overlay: Adds text to an image according to specified parameters such as alignment, color, font size, etc.
# PromptSelector and PromptSchedule
These classes are used to handle and format text prompts. They support selecting prompts based on input indices and applying pre- and post-text to formatted outputs.

# Loop, LoopStart, LoopEnd
A set of classes designed to facilitate custom loop constructs that can dynamically handle iteration based on runtime conditions.

# EndQueue
Conditioning end of current queue

## Contributions
Contributions are welcome. Please fork the repository and submit a pull request with your suggested changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.