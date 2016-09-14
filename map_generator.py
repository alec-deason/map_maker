from terrain import perlin_base_terrain, gradient_water
from visualize_terrain import visualize_water

CONFIG = {
        'base_config': {
            'seed': 100,
            'output_dir': '.',
        },
        'base_terrain': {
            'function': perlin_base_terrain,
            'config': {
                'width': 300,
                'height': 300,
                'frequency': 32,
                'octaves': 16,
                'material_coarseness': 2,
                'material_layers': 30,
                'max_elevation': 100000,
                }
        },
        'terrain_post_processors': [
            {
                'name': 'Hydrolic Erosion',
                'function': gradient_water,
                'config': {
                    'iterations': 130,
                    'rain_rate': 150,
                    'water_line_percentile': 20,
                },
            },
        ],
        'output_functions': [
            {
                'name': 'Draw Water',
                'function': visualize_water,
            },
        ],
        }

def main():
    base_config = CONFIG['base_config']

    ctx = {}

    print('Calculatinng base terrain...')
    ctx = CONFIG['base_terrain']['function'](ctx, dict(base_config, **CONFIG['base_terrain'].get('config', {})))
    for processor in CONFIG['terrain_post_processors']:
        print('Running post-processor {}'.format(processor['name']))
        ctx = processor['function'](ctx, dict(base_config, **processor.get('config', {})))
    print('Writing output to {}'.format(CONFIG['base_config']['output_dir']))
    for output in CONFIG['output_functions']:
        output['function'](ctx, dict(base_config, **output.get('config', {})))

if __name__ == '__main__':
    main()
