import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser(description="Artificial vision algorithm repository")
ap.add_argument("-method", metavar='algoritmo', required=True, help="metodo que se quiera ejecutar")
ap.add_argument("image", metavar='imagePath', type=str, help='Path to image yo want to process')
ap.add_argument("inRange", metavar='[intensityLevel]', type=int, nargs=2, help="Rango de niveles de intensidad")

# ap.add_argument("-t", "--test", required=True, help="path to output directory of images")
args = vars(ap.parse_args())

if args['method'] == 'adjustIntensity':
    print("adjusting")

if args['method'] == 'equalizeIntensity':
    print("Not implemented yet!")
