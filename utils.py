
houses = {
    'Gryffindor',
    'Hufflepuff',
    'Ravenclaw',
    'Slytherin'
}


def get_house_color(house: str):
    """ Returns the color associated with a Hogwarts house.
    """
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    }

    if house not in colors.keys():
        raise ValueError('Invalid house')

    return colors[house]
