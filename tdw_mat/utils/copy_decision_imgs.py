import re
import os
import shutil

def parse_and_copy_images(log_content: str, source_root: str, dest_root: str):
    """
    Parses a log string, constructs image/map paths, and copies them
    to a new directory.

    Args:
        log_content: A string containing the entire log file content.
        source_root: The root directory where source images are located.
        dest_root: The destination directory for the copied files.
    """
    # Regex to capture agent_name, frame number, and step number
    log_pattern = re.compile(r"agent_name: (Alice|Bob):LLM plan:.*? at frame (\d+), step (\d+)")
    
    copied_files = set()
    
    print(f"Starting file copy process to '{dest_root}'...")

    for line in log_content.strip().split('\n'):
        match = log_pattern.search(line)
        if not match:
            continue

        agent_name, frame_str, step_str = match.groups()
        frame, step = int(frame_str), int(step_str)
        agent_id = 0 if agent_name == "Alice" else 1
        step = step - 1  # Adjust step to be zero-indexed
        # Define source and destination paths for all three file types
        files_to_copy = [
            # (source_path, destination_path)
            (
                os.path.join(source_root, f"Images/{agent_id}", f"{step:04d}_{frame:04d}.png"),
                os.path.join(dest_root, f"Images/{agent_id}", f"{step:04d}_{frame:04d}.png")
            ),
            (
                os.path.join(source_root, f"Images/{agent_id}", f"{step:04d}_map.png"),
                os.path.join(dest_root, f"Images/{agent_id}", f"{step:04d}_map.png")
            ),
            (
                os.path.join(source_root, "top_down_image", f"img_{frame:05d}.jpg"),
                os.path.join(dest_root, "top_down_image", f"img_{frame:05d}.jpg")
            )
        ]

        for src, dest in files_to_copy:
            if dest in copied_files:
                continue

            dest_dir = os.path.dirname(dest)
            os.makedirs(dest_dir, exist_ok=True)

            try:
                if os.path.exists(src):
                    shutil.copy2(src, dest)
                    print(f"  [SUCCESS] Copied '{src}' to '{dest}'")
                    copied_files.add(dest)
                else:
                    print(f"  [SKIPPED] Source file not found: '{src}'")
            except Exception as e:
                print(f"  [ERROR] Failed to copy '{src}': {e}")
                
    print(f"\nFile copy process finished. Total unique files copied: {len(copied_files)}")

def setup_dummy_source_files(log_content: str, source_root: str):
    """
    Creates empty files to simulate the source directory structure.
    !!! FOR DEMONSTRATION/TESTING PURPOSES ONLY !!!
    """
    print("--- Setting up dummy source files for demonstration ---")
    log_pattern = re.compile(r"agent_name: (Alice|Bob):LLM plan:.*? at frame (\d+), step (\d+)")
    
    os.makedirs(os.path.join(source_root, "Images/0"), exist_ok=True)
    os.makedirs(os.path.join(source_root, "Images/1"), exist_ok=True)
    os.makedirs(os.path.join(source_root, "top_down_image"), exist_ok=True)

    created_files = set()
    for line in log_content.strip().split('\n'):
        match = log_pattern.search(line)
        if not match:
            continue
        agent_name, frame_str, step_str = match.groups()
        frame, step, agent_id = int(frame_str), int(step_str), (0 if agent_name == "Alice" else 1)
        step = step - 1
        files_to_create = [
            os.path.join(source_root, f"Images/{agent_id}", f"{step:04d}_{frame:04d}.png"),
            os.path.join(source_root, f"Images/{agent_id}", f"{step:04d}_map.png"),
            os.path.join(source_root, "top_down_image", f"{frame:04d}.jpg")
        ]
        for f_path in files_to_create:
            if f_path not in created_files:
                with open(f_path, 'w') as f: pass
                created_files.add(f_path)
    print(f"--- Created {len(created_files)} dummy files in '{source_root}/' ---\n")


if __name__ == "__main__":
    # Define constants for directories and the log filename
    episode_path = "results_coela/LMs-deepseek-chat/run_1"
    episode = "7"
    SOURCE_DIRECTORY = os.path.join(episode_path,episode)  # This is where your 'Images' and 'top_down_image' folders are
    DESTINATION_DIRECTORY = os.path.join(episode_path,episode, f"plan_imgs_{episode}")     # This is where the results will be saved
    LOG_FILENAME = os.path.join(episode_path,episode, f"llm_plan_{episode}.log")

    # --- Step 1: Read the log file ---
    try:
        with open(LOG_FILENAME, 'r', encoding='utf-8') as f:
            log_data = f.read()
        print(f"Successfully loaded log data from '{LOG_FILENAME}'.")
    except FileNotFoundError:
        print(f"[FATAL ERROR] Log file not found at '{os.path.join(os.getcwd(), LOG_FILENAME)}'")
        print("Please make sure 'plan.log' is in the same directory as the script.")
        exit()  # Exit the script if the log file is not found
    except Exception as e:
        print(f"[FATAL ERROR] An error occurred while reading the log file: {e}")
        exit()

    # --- Step 2 (Optional): Setup dummy files for testing ---
    # This function creates an empty file for every path it finds in the log.
    # In your actual use case with real images, you can DELETE or COMMENT OUT this line.
    # setup_dummy_source_files(log_data, SOURCE_DIRECTORY)

    # --- Step 3: Run the main function to parse the log and copy the files ---
    parse_and_copy_images(log_data, SOURCE_DIRECTORY, DESTINATION_DIRECTORY)