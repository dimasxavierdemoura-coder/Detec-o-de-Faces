##### Sistema de detecção de faces humanas
##### Todos os arquivos devem ser fornecidos localmente pelo usuário

import argparse
import os
import sys

import numpy as np
import cv2

##### Função para validar que o caminho informado existe e é um arquivo

def validate_file_path(path, description):
    if path is None or path.strip() == '':
        raise ValueError(f"{description} não foi informado.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} não encontrado: {path}")

    if not os.path.isfile(path):
        raise ValueError(f"{description} informado não é um arquivo: {path}")

    return os.path.abspath(path)

##### Função para carregar a imagem de disco

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Falha ao ler a imagem: {image_path}")
    return image

##### Função para carregar o modelo Caffe localmente

def load_face_detector(prototxt_path, model_path):
    return cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

##### Função para redimensionar a imagem preservando proporção

def resize_image(image, width=400):
    (h, w) = image.shape[:2]
    ratio = width / float(w)
    new_dimensions = (width, int(h * ratio))
    return cv2.resize(image, new_dimensions)

##### Função que aplica a detecção de rosto no OpenCV DNN

def detect_faces(image, net, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = f"{confidence * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return image

##### Função para encontrar o modelo Caffe local no diretório rest

def find_default_model(base_dir):
    rest_dir = os.path.join(base_dir, 'rest')
    candidates = [
        'res10_300x300_ssd_iter_140000.caffemodel',
        'res10_300x300_ssd_iter_140000_fp16.caffemodel',
    ]

    for name in candidates:
        candidate_path = os.path.join(rest_dir, name)
        if os.path.isfile(candidate_path):
            return os.path.abspath(candidate_path)

    if os.path.isdir(rest_dir):
        for file_name in os.listdir(rest_dir):
            if file_name.lower().endswith('.caffemodel'):
                return os.path.abspath(os.path.join(rest_dir, file_name))

    return None

##### Função para encontrar o prototxt padrão no diretório deploy

def find_default_prototxt(base_dir):
    deploy_dir = os.path.join(base_dir, 'deploy')
    default_path = os.path.join(deploy_dir, 'deploy.prototxt')
    if os.path.isfile(default_path):
        return os.path.abspath(default_path)

    if os.path.isdir(deploy_dir):
        for file_name in os.listdir(deploy_dir):
            if file_name.lower().endswith('.prototxt'):
                return os.path.abspath(os.path.join(deploy_dir, file_name))

    return None

##### Função para obter todas as imagens válidas de um diretório

def get_image_paths_from_directory(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Diretório não encontrado: {directory}')

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = [
        os.path.join(directory, file_name)
        for file_name in sorted(os.listdir(directory))
        if os.path.splitext(file_name)[1].lower() in valid_extensions
    ]

    if not image_paths:
        raise FileNotFoundError(f'Nenhuma imagem válida encontrada em: {directory}')

    return image_paths

##### Função para salvar a imagem de saída

def save_image(image, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not cv2.imwrite(output_path, image):
        raise IOError(f'Falha ao salvar a imagem em: {output_path}')

    print(f'[INFO] Resultado salvo em: {output_path}')

##### Função para ler argumentos de linha de comando

def parse_arguments(default_prototxt=None, default_model=None):
    parser = argparse.ArgumentParser(
        description='Sistema de detecção de faces humanas usando arquivos locais')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--image',
                       help='Caminho da imagem para detectar rostos')
    group.add_argument('-d', '--input-dir',
                       help='Diretório contendo imagens para processar em lote')

    parser.add_argument('-p', '--prototxt', default=default_prototxt,
                        help='Caminho para o arquivo deploy.prototxt do modelo')
    parser.add_argument('-m', '--model', default=default_model,
                        help='Caminho para o arquivo de pesos Caffe (.caffemodel)')
    parser.add_argument('-c', '--confidence', type=float, default=0.5,
                        help='Limite mínimo de confiança para aceitar uma detecção')
    parser.add_argument('-w', '--width', type=int, default=400,
                        help='Largura para redimensionamento da imagem antes da detecção')
    parser.add_argument('-o', '--output', help='Caminho de saída da imagem ou diretório para lotes')
    parser.add_argument('--no-display', action='store_true',
                        help='Não exibir a janela de resultado')

    return parser.parse_args()

##### Função para solicitar entrada do usuário no terminal

def prompt_input(text, default=None):
    while True:
        if default:
            prompt = f"{text} [{default}]: "
        else:
            prompt = f"{text}: "

        value = input(prompt).strip()
        if value:
            return value
        if default is not None:
            return default
        print('Valor obrigatório. Tente novamente.')


def prompt_for_file(text, default=None):
    while True:
        path = prompt_input(text, default)
        if os.path.isfile(path):
            return os.path.abspath(path)
        print(f'Arquivo não encontrado ou inválido: {path}')


def prompt_for_directory(text, default=None):
    while True:
        path = prompt_input(text, default)
        if os.path.isdir(path):
            return os.path.abspath(path)
        print(f'Diretório não encontrado ou inválido: {path}')

##### Função principal que orquestra o processamento

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_prototxt = find_default_prototxt(base_dir)
    default_model = find_default_model(base_dir)

    args = parse_arguments(default_prototxt=default_prototxt, default_model=default_model)

    if not args.image and not args.input_dir:
        print('[INFO] Nenhum argumento informado. Modo interativo ativado.')
        while True:
            choice = input('Processar (i)magem ou (d)iretório? [i/d]: ').strip().lower()
            if choice.startswith('d'):
                args.input_dir = prompt_for_directory('Digite o caminho do diretório de imagens')
                break
            elif choice.startswith('i') or choice == '':
                args.image = prompt_for_file('Digite o caminho da imagem')
                break
            else:
                print('Opção inválida. Digite "i" para imagem ou "d" para diretório.')

    if args.image and args.input_dir:
        print('[ERRO] Informe apenas uma opção: imagem (-i) ou diretório (-d).')
        sys.exit(1)

    if not args.prototxt:
        if default_prototxt:
            args.prototxt = default_prototxt
            print(f'[INFO] Usando prototxt padrão: {args.prototxt}')
        else:
            args.prototxt = prompt_input('Digite o caminho do arquivo deploy.prototxt')

    if not args.model:
        if default_model:
            args.model = default_model
            print(f'[INFO] Usando modelo padrão: {args.model}')
        else:
            args.model = prompt_input('Digite o caminho do arquivo de pesos Caffe (.caffemodel)')

    if args.image and not args.output:
        save_prompt = input('Deseja salvar o resultado em arquivo? [s/N]: ').strip().lower()
        if save_prompt.startswith('s'):
            args.output = prompt_input('Digite o caminho de saída para a imagem resultante')

    try:
        prototxt_path = validate_file_path(args.prototxt, 'Arquivo prototxt')
        model_path = validate_file_path(args.model, 'Arquivo de pesos do modelo')
    except Exception as e:
        print(f'[ERRO] {e}')
        if args.prototxt is None:
            print('[ERRO] Não foi possível encontrar deploy.prototxt em deploy/.')
        if args.model is None:
            print('[ERRO] Não foi possível encontrar um arquivo .caffemodel em rest/.')
        sys.exit(1)

    if args.image:
        image_paths = [validate_file_path(args.image, 'Arquivo de imagem')]
    else:
        image_paths = get_image_paths_from_directory(args.input_dir)

    if args.input_dir:
        if args.output:
            output_dir = os.path.abspath(args.output)
            if os.path.exists(output_dir) and not os.path.isdir(output_dir):
                print(f'[ERRO] O caminho de saída para lote deve ser um diretório: {output_dir}')
                sys.exit(1)
            if not os.path.exists(output_dir) and os.path.splitext(output_dir)[1]:
                print(f'[ERRO] Para processamento em lote, --output deve ser um diretório, não um arquivo: {output_dir}')
                sys.exit(1)
        else:
            output_dir = os.path.join(args.input_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    print('[INFO] Carregando o modelo de detecção de rosto local...')
    net = load_face_detector(prototxt_path, model_path)

    for image_path in image_paths:
        try:
            print(f'[INFO] Processando: {image_path}')
            image = load_image(image_path)
            image = resize_image(image, width=args.width)
            output_image = detect_faces(image, net, args.confidence)

            if args.image and args.output:
                output_path = args.output
                if os.path.isdir(output_path):
                    file_name = os.path.basename(image_path)
                    name, ext = os.path.splitext(file_name)
                    output_path = os.path.join(output_path, f'{name}_faces{ext}')
                save_image(output_image, output_path)
            elif args.input_dir:
                file_name = os.path.basename(image_path)
                name, ext = os.path.splitext(file_name)
                output_file = os.path.join(output_dir, f'{name}_faces{ext}')
                save_image(output_image, output_file)

            if not args.no_display:
                cv2.imshow('Detecção de Faces', output_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            print(f'[ERRO] Não foi possível processar {image_path}: {e}')

    print('[INFO] Processamento concluído.')


if __name__ == '__main__':
    main()
 
