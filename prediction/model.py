from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Embedding
from tensorflow.keras.utils import plot_model
import numpy as np


def seq2seq_ext(latent_dim, num_encoder_tokens, num_decoder_tokens, ext_quant_dim, ext_cate_dim_list):
    # Define an input sequence and process it.
    encoder_inputs = []
    outputs = []
    inp_main = Input(shape=(None, num_encoder_tokens), name='en_prev_flow')
    encoder_inputs.append(inp_main)
    outputs.append(inp_main)
    inp_quant = Input(shape=(None, ext_quant_dim), name='en_ext_quant')
    encoder_inputs.append(inp_quant)
    outputs.append(inp_quant)
    i = 0
    for ext_cate_dim in ext_cate_dim_list:
        inp_cate = Input(shape=(None,), name='en_ext_cate_{}'.format(i))
        encoder_inputs.append(inp_cate)
        output = Embedding(input_dim=ext_cate_dim, output_dim=int(np.sqrt(ext_cate_dim)))(inp_cate)
        outputs.append(output)
        i += 1
    out = Concatenate()(outputs)
    fused = Dense(units=8, activation='relu')(out)
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(fused)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    decoder_inputs = []
    outputs = []
    inp_main = Input(shape=(None, num_decoder_tokens), name='de_prev_flow')
    decoder_inputs.append(inp_main)
    outputs.append(inp_main)
    inp_quant = Input(shape=(None, ext_quant_dim), name='de_external_quant')
    decoder_inputs.append(inp_quant)
    outputs.append(inp_quant)
    i = 0
    for ext_cate_dim in ext_cate_dim_list:
        inp_cate = Input(shape=(None,), name='de_external_cate_{}'.format(i))
        decoder_inputs.append(inp_cate)
        output = Embedding(ext_cate_dim, int(np.sqrt(ext_cate_dim)))(inp_cate)
        outputs.append(output)
        i += 1
    out = Concatenate()(outputs)
    fused = Dense(units=8, activation='relu')(out)

    # Set up the decoder, using `encoder_states` as initial state.
    # decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(fused,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens)
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    seq2seq_model = Model(encoder_inputs + decoder_inputs, decoder_outputs)

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        fused, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        decoder_inputs + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    return seq2seq_model, encoder_model, decoder_model


if __name__ == '__main__':
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

    seq2seq_model, encoder_model, decoder_model = seq2seq_ext(8, 1, 1, 5, [8, 8, 24])
    plot_model(seq2seq_model, to_file='seq2seq_ext_model.png', show_shapes=True)
    plot_model(encoder_model, to_file='encoder_ext_model.png', show_shapes=True)
    plot_model(decoder_model, to_file='decoder_ext_model.png', show_shapes=True)
    seq2seq_model.summary()
