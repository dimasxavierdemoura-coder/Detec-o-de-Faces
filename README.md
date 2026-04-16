Bem-vindo ao sistema de detecção de faces humanas.

Aqui está tudo pronto: o modelo, o prototxt e o script já estão incluídos no repositório. O código foi pensado para ser simples de usar e confiável, com suporte a imagem única ou processamento em lote.

Arquivos principais:
- `sistema_detecta_face.py`
- `deploy/deploy.prototxt`
- `rest/res10_300x300_ssd_iter_140000.caffemodel`

Como executar:
python "sistema_detecta_face.py" ^
 entre com as informções de processamento individual ou em lote
 entre com as informções de localização


O código já cuida de criar a pasta de saída quando necessário, e o projeto está pronto para uso imediato.
