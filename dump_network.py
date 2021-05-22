import json


# Reference: https://devtalk.nvidia.com/default/topic/1064669/tensorrt/troubleshooting-suggestions-for-onnx-v-tensorrt-discrepancies/post/5392296/#5392296

def dump_network(network, filename):
    """Dumps TensorRT parsed network attributes to a JSON file.



    Parameters

    ----------

    network: tensorrt.INetworkDefinition

        Network created from parsing original model (ONNX, etc.)



    filename: str

        Filename to dump the network info to in JSON format.



    Example

    -------

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(), trt.OnnxParser(network, TRT_LOGGER) as parser:

        ...



        # Fill network attributes with information by parsing model

        with open("model.onnx", "rb") as f:

            parser.parse(f.read())



        dump_network(network, "network.json")

    """

    network_description = {}

    for i in range(network.num_layers):

        layer = network.get_layer(i)

        network_description[i] = {

            a: getattr(layer, a).__repr__() for a in

            ('name', 'type', 'precision', 'precision_is_set', 'num_inputs',

             'num_outputs')

        }

        network_description[i]['inputs'] = {

            i: {

                a: getattr(layer.get_input(i), a).__repr__() for a in

                ('name', 'shape', 'dtype')

            } for i in range(layer.num_inputs)

        }

        network_description[i]['outputs'] = {

            i: {

                a: getattr(layer.get_output(i), a).__repr__() for a in

                ('name', 'shape', 'dtype')

            } for i in range(layer.num_outputs)

        }

    with open(filename, 'w') as fp:

        print("Writing {:}".format(filename))

        json.dump(network_description, fp, indent=4, sort_keys=True)
