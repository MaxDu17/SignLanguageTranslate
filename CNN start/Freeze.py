from tensorflow.python.tools import freeze_graph


input_graph_path = 'Graphs_and_Results/graph.pbtxt'



checkpoint_path = 'Graphs_and_Results/CNN_test-100'
input_saver_def_path = ''
input_binary = False
output_node_names = 'Output/Prediction'
restore_op_name = 'save/restore_all'
filename_tensor_name = 'save/Const:0'
output_frozen_graph_name = 'Graphs_and_Results/CNN_test.pb'

clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, '')

print("I'm done.")
