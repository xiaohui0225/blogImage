import onnx
import onnx.helper as helper


model = onnx.load("yolov5s.onnx")


def find_node_for_output(nodes, name):
	for i, n in enumerate(nodes):
		if name in n.output:
			return i, n
	return None, None

nodes = model.graph.node
inits = model.graph.initializer
remove_nodes = []

for i, node in enumerate(nodes):
	if node.op_type == "Resize":
		idx, identity = find_node_for_output(nodes, node.input[2])
		if identity is not None:
			remove_nodes.append(idx)
			node.input[2] = identity.input[0]

remove_nodes = sorted(remove_nodes,reverse=True)
for i in remove_nodes:
	del nodes[i]

onnx.save(model, "output.onnx")