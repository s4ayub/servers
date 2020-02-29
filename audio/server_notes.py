# Formulate sliding window
window_size = round(window_size_ms * (fs/1000))
window_step = round(window_step_ms * (fs/1000))


for x in range(0, len(chunk), window_step):
    window_data = chunk[x:x+window_size]
    output_file = os.path.join(output_subfolder, input_filename)
    image_path = '%s_%d_%d.png' % (output_file, chunk_timestamp, m * window_step_ms)
