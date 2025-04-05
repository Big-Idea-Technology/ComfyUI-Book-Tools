#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
import numpy as np
import random
import math

class BookToolsTextToImage:
    def __init__(self, device="gpu"):
        self.device = device
    _positions = ["top", "bottom"]
    _alignments = ["left", "center"]
    _colors = [

    (255, 50, 50),      
    (0, 150, 255),      
    (255, 225, 25),     

    (255, 150, 50),     
    (100, 220, 100),    
    (180, 80, 220),     
    (0, 210, 210),      

    (255, 170, 200),    
    (150, 220, 255),    
    (170, 255, 170),    
    (255, 220, 150),    
    (220, 180, 255),    

    (255, 100, 100),    
    (100, 200, 100),    
    (255, 180, 50),     
    (80, 150, 255),     
    (220, 100, 220),    

    (255, 255, 255),    
    (240, 240, 0),      
    (0, 220, 130),      
    (255, 100, 180),    

    (255, 200, 0),      
    (200, 100, 255),    
    (100, 220, 220),    
    (255, 150, 150)     
]

    def font_with_weight(self, font_path):
        try:
            if "VariableFont" in font_path:
                from fontTools.ttLib import TTFont
                font = TTFont(font_path)
                if 'fvar' in font:
                    font_obj = ImageFont.truetype(font_path, size=12)
                    if hasattr(font_obj, 'set_variation_by_axes'):
                        font_obj.set_variation_by_axes([700])  # Set weight to bold (700)
                    return font_path
            return font_path
        except Exception as e:
            print(f"Variable font weight setting failed: {e}")
            return font_path

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "text": ("STRING", {"multiline": True, "default": "Hello"}),
                "max_font_size": ("INT", {"default": 96, "min": 8, "max": 256}),
                "min_font_size": ("INT", {"default": 32, "min": 8, "max": 256}),
                "font": ("STRING", {"default": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"}),
                "padding": ("INT", {"default": 40, "min": 0}),
                "line_height_factor": ("FLOAT", {"default": 1.4, "min": 0.5, "max": 3.0, "step": 0.1}),
                "curve_amount": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.1}),
                "char_spacing": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1}),
                "random_position": ("BOOLEAN", {"default": False}),
                "position_offset": ("INT", {"default": 20, "min": 0}),
                "shadow_offset": ("INT", {"default": 4, "min": 0, "max": 20}),
                "outline_thickness": ("INT", {"default": 3, "min": 1, "max": 10}),
                "use_gradient": ("BOOLEAN", {"default": False}),
                "gradient_direction": (["vertical", "horizontal"], {"default": "vertical"}),
                "background_style": (["none", "rectangle", "ribbon", "glow"], {"default": "none"}),
                "background_color": (["auto", "dark", "light"], {"default": "auto"}),
                "background_padding": ("INT", {"default": 20, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_text_image"
    CATEGORY = "image/text"

    def calculate_text_width(self, text, font):
        """Calculate exact text width including kerning if available"""
        try:
            # Try to use layout engine if available (more accurate)
            dummy_image = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(dummy_image)
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0]
        except:
            # Fallback to simple width calculation
            return sum(font.getbbox(char)[2] - font.getbbox(char)[0] for char in text)

    def calculate_text_size(self, text, font_size, font_path, max_width, max_height, line_height_factor):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Font error: {str(e)}, using default")
            font = ImageFont.load_default()
            
        paragraphs = text.split('\n')
        lines = []
        
        max_line_width = 0
        
        for paragraph in paragraphs:
            words = paragraph.split()
            current_line = []
            current_width = 0
            
            # Consider character spacing in the width calculation
            char_spacing = font_size * 0.2  # Default char_spacing from parameters
            
            for word in words:
                # Get word width including spacing between characters
                word_width = 0
                for char in word:
                    bbox = font.getbbox(char)
                    word_width += (bbox[2] - bbox[0]) + char_spacing
                
                # Remove extra spacing after the last character
                if word:
                    word_width -= char_spacing
                
                # Consider space width
                space_width = font.getbbox(' ')[2] if current_line else 0
                
                if current_width + word_width + space_width <= max_width:
                    current_line.append(word)
                    current_width += word_width + space_width
                else:
                    if current_line:
                        line_text = ' '.join(current_line)
                        lines.append(line_text)
                        line_width = 0
                        for char in line_text:
                            bbox = font.getbbox(char)
                            line_width += (bbox[2] - bbox[0]) + char_spacing
                        max_line_width = max(max_line_width, line_width)
                    
                    current_line = [word]
                    current_width = word_width
            
            if current_line:
                line_text = ' '.join(current_line)
                lines.append(line_text)
                line_width = 0
                for char in line_text:
                    bbox = font.getbbox(char)
                    line_width += (bbox[2] - bbox[0]) + char_spacing
                max_line_width = max(max_line_width, line_width)
        
        line_height = int(font_size * line_height_factor)
        total_height = len(lines) * line_height
        
        # Check if the text fits both width and height constraints
        fits = (max_line_width <= max_width) and (total_height <= max_height)
        
        return lines, fits, max_line_width, total_height

    def create_gradient_color(self, color, height, direction="vertical"):
        start_color = color
        end_color = tuple(max(0, c - 80) for c in color)  # Darker version
        gradient_colors = []
        
        for i in range(height):
            if direction == "vertical":
                ratio = i / height
            else:
                ratio = 0.5  # For horizontal, we'll handle it differently
            
            gradient_color = tuple(
                int(start_color[j] + (end_color[j] - start_color[j]) * ratio)
                for j in range(3)
            )
            gradient_colors.append(gradient_color)
        return gradient_colors

    def draw_background(self, draw, x, y, width, height, style, color_mode, base_color):
        if style == "none":
            return

        bg_color = self.get_background_color(color_mode, base_color)
        
        # Create a separate image for the background with blur
        bg_img = Image.new('RGBA', draw._image.size, (0,0,0,0))
        bg_draw = ImageDraw.Draw(bg_img)
        
        if style == "rectangle":
            # Draw rounded rectangle with shadow
            shadow_offset = 8
            shadow_color = (0, 0, 0, 100)
            self.rounded_rectangle(bg_draw, 
                            (x + shadow_offset, y + shadow_offset, 
                             x + width + shadow_offset, y + height + shadow_offset),
                            fill=shadow_color, radius=20)
            self.rounded_rectangle(bg_draw, (x, y, x + width, y + height),
                            fill=bg_color, radius=20)
            
        elif style == "ribbon":
            # Draw ribbon effect
            points = [
                (x, y + height/2),
                (x + 20, y),
                (x + width - 20, y),
                (x + width, y + height/2),
                (x + width - 20, y + height),
                (x + 20, y + height)
            ]
            bg_draw.polygon(points, fill=bg_color)
            
        elif style == "glow":
            # Create radial gradient glow with reduced opacity
            for i in range(20, 0, -1):
                # Reduce overall opacity to 60% of original
                alpha = int(255 * 0.6 * (1 - i/20))
                glow_color = (*bg_color[:3], alpha)
                expanded_rect = (
                    x - i, y - i,
                    x + width + i, y + height + i
                )
                self.rounded_rectangle(bg_draw, expanded_rect, fill=glow_color, radius=20+i)

        # Apply gaussian blur to background
        bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=3))
        # Paste blurred background onto main image
        draw._image.paste(bg_img, (0,0), bg_img)

    def get_background_color(self, mode, text_color):
        if mode == "auto":
            # Create contrasting color
            brightness = sum(text_color[:3])/3
            return (30, 30, 30, 180) if brightness > 128 else (225, 225, 225, 180)
        elif mode == "dark":
            return (30, 30, 30, 180)
        else:  # light
            return (225, 225, 225, 220)

    def draw_text_with_border(self, draw, text, font, x, y, char_spacing, fill_color=None, 
                             shadow_offset=4, outline_thickness=3, use_gradient=False, 
                             gradient_direction="vertical", background_style="none", 
                             background_color="auto", background_padding=20):
        if fill_color is None:
            fill_color = random.choice(self._colors)

        # Calculate text dimensions including spacing
        total_width = 0
        for i, char in enumerate(text):
            bbox = font.getbbox(char)
            char_width = bbox[2] - bbox[0]
            total_width += char_width
            if i < len(text) - 1:  # Add spacing between chars
                total_width += font.size * char_spacing

        # Get text height
        text_height = font.getbbox(text)[3] - font.getbbox(text)[1]
        
        # Draw background if needed
        if background_style != "none":
            bg_x = x - background_padding
            bg_y = y - background_padding
            bg_width = total_width + (background_padding * 2)  # Now includes char spacing
            bg_height = text_height + (background_padding * 2)
            self.draw_background(draw, bg_x, bg_y, bg_width, bg_height,
                               background_style, background_color, fill_color)

        # Calculate spacing and dimensions
        spacing = font.size * char_spacing
        text_height = font.getbbox(text)[3] - font.getbbox(text)[1]
        
        # Calculate text dimensions first
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create gradient colors if needed
        gradient_colors = self.create_gradient_color(fill_color, text_height, gradient_direction) if use_gradient else None

        # Draw shadow first (offset version)
        shadow_color = (30, 30, 30)
        current_x = x
        for i, char in enumerate(text):
            bbox = font.getbbox(char)
            width = bbox[2] - bbox[0]
            
            # Draw shadow with increasing offset
            shadow_y_offset = shadow_offset + (i % 2)  # Slightly varied shadow
            draw.text((current_x + shadow_offset, y + shadow_y_offset), char, font=font, fill=shadow_color)
            
            current_x += width + (spacing if i < len(text) - 1 else 0)

        # Draw outline and main text
        current_x = x
        for i, char in enumerate(text):
            bbox = font.getbbox(char)
            width = bbox[2] - bbox[0]
            
            # Enhanced outline with varied thickness
            for dx, dy in [(i,j) for i in range(-outline_thickness, outline_thickness+1) 
                          for j in range(-outline_thickness, outline_thickness+1)
                          if (i*i + j*j) <= outline_thickness*outline_thickness]:
                draw.text((current_x+dx, y+dy), char, font=font, fill=(0,0,0))
            
            if use_gradient:
                # Create temporary image for this character
                char_img = Image.new('RGBA', (width, text_height), (0,0,0,0))
                char_draw = ImageDraw.Draw(char_img)
                
                # Draw character in white to create mask
                char_draw.text((0, 0), char, font=font, fill=(255,255,255,255))
                
                # Create gradient image
                gradient_img = Image.new('RGBA', (width, text_height), (0,0,0,0))
                for h in range(text_height):
                    color = gradient_colors[h]
                    ImageDraw.Draw(gradient_img).line([(0, h), (width, h)], fill=(*color, 255))
                
                # Use character as mask for gradient
                char_mask = char_img.split()[3]  # Get alpha channel
                gradient_img.putalpha(char_mask)
                
                # Paste the gradient character
                draw._image.paste(gradient_img, (int(current_x), int(y)), char_mask)
            else:
                # Regular colored text
                draw.text((current_x, y), char, font=font, fill=fill_color)
            
            current_x += width + (spacing if i < len(text) - 1 else 0)

    def draw_curved_text(self, draw, text, font, x, y, width, curve_amount, char_spacing, text_color=None):
        # Calculate the maximum width available for text
        usable_width = min(width * 0.9, draw._image.width - 100)
        
        # Calculate spacing
        spacing = font.size * char_spacing
        
        # Calculate character widths and total width without extra spacing at end
        char_widths = []
        total_natural_width = 0
        for i, char in enumerate(text):
            bbox = font.getbbox(char)
            char_width = bbox[2] - bbox[0]
            char_widths.append(char_width)
            total_natural_width += char_width
            # Only add spacing between characters, not after the last one
            if i < len(text) - 1:
                total_natural_width += spacing

        # Set starting position correctly based on total width
        start_x = x - total_natural_width/2
        current_x = start_x

        if text_color is None:
            text_color = random.choice(self._colors)

        for i, (char, char_width) in enumerate(zip(text, char_widths)):
            # Calculate character center and progress
            char_center_x = current_x + char_width/2
            progress = (char_center_x - x) / (usable_width/2)
            progress = max(-1, min(1, progress))  # Clamp progress
            
            offset_y = curve_amount * width * (progress * progress)
            
            # Create character image with padding
            padding = int(font.size * 0.2)
            char_img = Image.new('RGBA', (int(char_width + padding*2), int(font.size*2)), (0,0,0,0))
            char_draw = ImageDraw.Draw(char_img)
            
            # Center character in its image
            char_x = padding
            char_y = char_img.height/2 - font.size/2
            
            # Draw border
            thickness = max(2, font.size // 25)
            for dx, dy in [(-thickness,-thickness), (thickness,-thickness), (-thickness,thickness), (thickness,thickness),
                          (-thickness,0), (thickness,0), (0,-thickness), (0,thickness)]:
                char_draw.text((char_x+dx, char_y+dy), char, font=font, fill=(0,0,0,255))
            
            # Draw character
            char_draw.text((char_x, char_y), char, font=font, fill=(*text_color, 255))
            
            # Calculate and clamp rotation angle
            angle = math.degrees(math.atan2(2 * curve_amount * width * progress, width/2))
            angle = max(-30, min(30, angle))  # Limit rotation more
            rotated = char_img.rotate(angle, expand=True, resample=Image.BICUBIC)
            
            # Calculate paste position
            paste_x = int(current_x - rotated.width/2)
            paste_y = int(y + offset_y - rotated.height/2)
            
            # Ensure within bounds
            paste_x = max(0, min(draw._image.width - rotated.width, paste_x))
            paste_y = max(0, min(draw._image.height - rotated.height, paste_y))
            
            mask = rotated.split()[3]
            draw._image.paste(rotated, (paste_x, paste_y), mask)
            
            # Move to next character position with consistent spacing
            current_x += char_width
            if i < len(text) - 1:  # Only add spacing if not the last character
                current_x += spacing

    def create_text_image(self, image, text, max_font_size, min_font_size, font, padding, 
                         line_height_factor, curve_amount=0.0, char_spacing=0.2, 
                         random_position=False, position_offset=20, shadow_offset=4,
                         outline_thickness=3, use_gradient=False, gradient_direction="vertical",
                         background_style="none", background_color="auto", background_padding=20):
        image_tensor = image
        image_np = image_tensor.cpu().numpy()
        image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))
        width, height = image.size
        
        draw = ImageDraw.Draw(image)
        font = self.font_with_weight(font)

        try:
            # Calculate effective width and height considering padding
            effective_width = width - (2 * padding)
            effective_height = height - (2 * padding)
            
            # Binary search to find optimal font size
            low, high = min_font_size, max_font_size
            optimal_font_size = min_font_size  # Initialize with min_font_size
            optimal_lines = []
            
            # First check if min_font_size fits
            lines, fits, actual_width, actual_height = self.calculate_text_size(
                text, min_font_size, font, effective_width, effective_height, line_height_factor
            )
            
            if fits:
                # Use binary search to find the largest font size that fits
                while low <= high:
                    mid = (low + high) // 2
                    lines, fits, actual_width, actual_height = self.calculate_text_size(
                        text, mid, font, effective_width, effective_height, line_height_factor
                    )
                    
                    if fits:
                        optimal_font_size = mid
                        optimal_lines = lines
                        low = mid + 1
                    else:
                        high = mid - 1
            else:
                # Even min_font_size doesn't fit, use it anyway with overflow
                optimal_font_size = min_font_size
                optimal_lines = lines
            
            loaded_font = ImageFont.truetype(font, optimal_font_size)
            
        except Exception as e:
            print(f"Warning: Could not load font {font}: {str(e)}, using default")
            loaded_font = ImageFont.load_default()
            optimal_lines = [text.upper()]

        line_height = int(optimal_font_size * line_height_factor)
        total_height = len(optimal_lines) * line_height

        if random_position:
            vertical_pos = random.choice(self._positions)
            horizontal_pos = random.choice(self._alignments)
            
            if vertical_pos == "top":
                base_y = padding + position_offset
            elif vertical_pos == "bottom":
                base_y = height - total_height - padding - position_offset
            else:
                base_y = (height - total_height) // 2
                
            base_y += random.randint(-10, 10)
        else:
            base_y = (height - total_height) // 2
            horizontal_pos = "center"

        y = base_y
        text_color = random.choice(self._colors)
        
        for line in optimal_lines:
            if curve_amount != 0:
                # For curved text, center is already handled in draw_curved_text
                self.draw_curved_text(draw, line, loaded_font, width//2, y+2, width - 2*padding, curve_amount, char_spacing, 
                                    text_color=(max(0, text_color[0] - 100),
                                              max(0, text_color[1] - 100),
                                              max(0, text_color[2] - 100)))
                self.draw_curved_text(draw, line, loaded_font, width//2, y, width - 2*padding, curve_amount, char_spacing, 
                                    text_color=text_color)
            else:
                # Calculate total line width including character spacing
                total_width = 0
                for i, char in enumerate(line):
                    bbox = loaded_font.getbbox(char)
                    char_width = bbox[2] - bbox[0]
                    total_width += char_width
                    if i < len(line) - 1:  # Add spacing between chars, but not after last char
                        total_width += loaded_font.size * char_spacing
                
                if horizontal_pos == "left":
                    x = padding + position_offset
                else:  # center
                    # Center based on actual total width including spacing
                    x = (width - total_width) // 2

                if random_position:
                    x += random.randint(-10, 10)
                
                self.draw_text_with_border(draw, line, loaded_font, x, y, char_spacing, text_color, shadow_offset, outline_thickness, use_gradient, gradient_direction, background_style, background_color, background_padding)
            
            y += line_height
        
        image_tensor_out = torch.tensor(np.array(image).astype(np.float32) / 255.0)
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
        return (image_tensor_out,)

    @staticmethod
    def rounded_rectangle(draw, bbox, fill=None, radius=20):
        x1, y1, x2, y2 = bbox
        
        # Ensure radius isn't too large for the rectangle
        width = x2 - x1
        height = y2 - y1
        radius = min(radius, width//2, height//2)
        
        if radius <= 0:
            draw.rectangle(bbox, fill=fill)
            return
            
        draw.rectangle((x1+radius, y1, x2-radius, y2), fill=fill)
        draw.rectangle((x1, y1+radius, x2, y2-radius), fill=fill)
        
        # Draw corners
        draw.pieslice((x1, y1, x1+radius*2, y1+radius*2), 180, 270, fill=fill)
        draw.pieslice((x2-radius*2, y1, x2, y1+radius*2), 270, 0, fill=fill)
        draw.pieslice((x1, y2-radius*2, x1+radius*2, y2), 90, 180, fill=fill)
        draw.pieslice((x2-radius*2, y2-radius*2, x2, y2), 0, 90, fill=fill)