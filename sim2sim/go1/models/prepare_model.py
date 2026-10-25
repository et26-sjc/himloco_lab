"""Reproduce the torque-only MJCF from an unmodified pinned Menagerie checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def prepare(source, destination, revision):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('go1.xml', 'scene.xml', 'LICENSE', 'README.md', 'CHANGELOG.md', 'go1.png'):
        shutil.copyfile(source/name, destination/name)
    shutil.copytree(source/'assets', destination/'assets', dirs_exist_ok=True)
    root = ET.parse(source/'go1.xml').getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'position':
                parent.remove(child)
    actuators = root.find('actuator')
    actuators.clear()
    for joint in root.findall('.//worldbody//joint'):
        name = joint.get('name')
        limit = 35.55 if '_calf_' in name else 23.7
        ET.SubElement(actuators, 'motor', name=name.removesuffix('_joint'), joint=name,
                      gear='1', ctrllimited='true', ctrlrange=f'{-limit} {limit}',
                      forcelimited='true', forcerange=f'{-limit} {limit}')
    # The upstream home key contains position actuator controls, not torques.
    keyframe = root.find('keyframe')
    if keyframe is not None:
        root.remove(keyframe)
    ET.indent(root)
    ET.ElementTree(root).write(destination/'go1_torque.xml', encoding='unicode')
    scene = ET.parse(source/'scene.xml').getroot()
    scene.find('include').set('file', 'go1_torque.xml')
    ET.indent(scene)
    ET.ElementTree(scene).write(destination/'scene_torque.xml', encoding='unicode')
    files = [p for p in destination.rglob('*') if p.is_file() and p.name != 'provenance.json']
    record = dict(repository='https://github.com/google-deepmind/mujoco_menagerie', commit=revision,
                  subdirectory='unitree_go1', license='BSD-3-Clause (see LICENSE)',
                  changes=['remove every position servo/default', 'replace 12 actuators with unit-gear torque motors',
                           'remove position-control home keyframe', 'torque scene includes go1_torque.xml'],
                  files={str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    (destination/'provenance.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', default=str(Path(__file__).parent/'unitree_go1'))
    args = parser.parse_args()
    prepare(args.source, args.output, args.revision)
