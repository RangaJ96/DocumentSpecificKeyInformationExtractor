from tensorflow import keras
from layers import GraphOperator, GNN, Adjacency
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import tensorflow
from tensorflow.keras.utils import plot_model
import re



tensorflow.executing_eagerly()
_VALID_SCOPE_NAME_REGEX = re.compile("^[A-Za-z0-9_.\\-/>]*$")


adjacency_input = keras.Input(shape=(50,50), batch_size=1, name='Adjacency_Input')
node_input = keras.Input(shape=(50, 5), batch_size=1, name='Node_Input')


node_embeddings = layers.Dense(units=50, name='Embedding')(node_input)
adj_list = GraphOperator()(adjacency_input)

adj_list_1 = [adj_tensor for adj_tensor in adj_list]
adj_list_1.append(node_embeddings)

adj_0_1, adj_1_1, adj_2_1 = Adjacency(n_features=50, max_nodes=50)(adj_list_1)
                                                        
gnn_1_1 = GNN(n_features=50, n_nodes=50)(inputs = [node_embeddings, adj_0_1, 
                                                    adj_1_1, adj_2_1])

bn_1_1 = layers.BatchNormalization(name='BN_1.1')(gnn_1_1)
relu_1 = layers.ReLU(name='ReLU_1')(bn_1_1)


gnn_2_1 = GNN(n_features=50, n_nodes=50)(inputs = [relu_1, adj_0_1, adj_1_1, 
                                                    adj_2_1])

bn_2_1 = layers.BatchNormalization(name='BN_1.2')(gnn_2_1)

add_1 = layers.add([bn_2_1, node_embeddings], name='Add_1')


adj_list_2 = [adj_tensor for adj_tensor in adj_list]
adj_list_2.append(add_1)


adj_0_2, adj_1_2, adj_2_2 = Adjacency(n_features=50, max_nodes=50)(adj_list_2) 
                                                                                                                     
gnn_1_2 = GNN(n_features=50, n_nodes=50)(inputs = [add_1, adj_0_2, adj_1_2, 
                                                            adj_2_2])

bn_1_2 = layers.BatchNormalization(name='BN_2.1')(gnn_1_2)
relu_2 = layers.ReLU(name='ReLU_2')(bn_1_2)

gnn_2_2 = GNN(n_features=50, n_nodes=50)(inputs = [relu_2, adj_0_2, 
                                                    adj_1_2, adj_2_2])

bn_2_2 = layers.BatchNormalization(name='BN_2.2')(gnn_2_2)

add_2 = layers.add([bn_2_2, add_1], name='Add_2')

adj_list_3 = [adj_tensor for adj_tensor in adj_list]
adj_list_3.append(add_2)

adj_0_3, adj_1_3, adj_2_3 = Adjacency(n_features=50, max_nodes=50)(adj_list_3)                                                            
                                                       
gnn_1_3 = GNN(n_features=50, n_nodes=50)(inputs = [add_2, adj_0_3, 
                                                    adj_1_3, adj_2_3])
bn_1_3 = layers.BatchNormalization(name='BN_3.1')(gnn_1_3)
relu_3 = layers.ReLU(name='ReLU_3')(bn_1_3)


gnn_2_3 = GNN(n_features=50, n_nodes=50)(inputs = [relu_3, adj_0_3, 
                                                    adj_1_3, adj_2_3])

bn_2_3 = layers.BatchNormalization(name='BN_3.2')(gnn_2_3)


add_3 = layers.add([bn_2_3, add_2], name='Add_3')



node_labels = layers.Dense(14, activation='sigmoid', name='Classifier')(add_3)




opt = SGD(lr = 0.001, momentum=0.9)

model = keras.Model(inputs=[adjacency_input, node_input], 
                    outputs=[node_labels])



plot_model(model, to_file='model.png', show_shapes=True, rankdir='TB')