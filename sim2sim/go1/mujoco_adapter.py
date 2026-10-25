"""Named joint mapping and the Lab policy contract from physical MuJoCo state."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .config import default_position
from .policy_runner import sha256


class MujocoAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        provenance_path = Path(cfg.model).parent/'provenance.json'
        if provenance_path.exists():
            for name, digest in json.loads(provenance_path.read_text())['files'].items():
                if sha256(provenance_path.parent/name) != digest:
                    raise ValueError(f'Model provenance checksum mismatch: {name}')
        self.model = mujoco.MjModel.from_xml_path(str(Path(cfg.model).resolve()))
        self.model.opt.timestep = cfg.physics_dt
        self.data = mujoco.MjData(self.model)
        if (self.model.nq, self.model.nv) != (19, 18):
            raise ValueError('Expected a floating Go1 with exactly twelve actuated joints')
        self.names = cfg.joint_names
        self.joint_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names])
        if not np.all(self.model.jnt_type[self.joint_ids] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError('Expected twelve hinge joints')
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.dof_ids = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = []
        for joint in self.joint_ids:
            ids = np.flatnonzero((self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                                 & (self.model.actuator_trnid[:, 0] == joint))
            if len(ids) != 1:
                raise ValueError('Expected exactly one torque motor per named joint')
            self.actuator_ids.append(int(ids[0]))
        self.actuator_ids = np.array(self.actuator_ids)
        # Reject authored position/velocity servos or any extra hidden actuator.
        if self.model.nu != 12 or not (
                np.all(self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
                and np.all(self.model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
                and np.all(self.model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE)
                and np.allclose(self.model.actuator_gainprm[:, 0], 1)
                and np.allclose(self.model.actuator_gainprm[:, 1:], 0)
                and np.allclose(self.model.actuator_biasprm, 0)
                and np.allclose(self.model.actuator_gear[:, 0], 1)
                and np.allclose(self.model.actuator_gear[:, 1:], 0)):
            raise ValueError('Model must contain only unit-gear torque motors, no position drive')
        self.default_pos = np.array([default_position(n) for n in self.names])
        self.effort_limits = np.array([35.55 if '_calf_' in n else 23.7 for n in self.names])
        expected = np.column_stack((-self.effort_limits, self.effort_limits))
        if not np.allclose(self.model.actuator_ctrlrange[self.actuator_ids], expected):
            raise ValueError('Motor ctrlrange must match Go1 joint effort limits')
        if (not np.all(self.model.actuator_ctrllimited) or not np.all(self.model.actuator_forcelimited)
                or not np.allclose(self.model.actuator_forcerange[self.actuator_ids], expected)):
            raise ValueError('Motor effort clipping must match Go1 joint effort limits')
        self.body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, 'trunk')
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1 or self.model.jnt_bodyid[free[0]] != self.body_id:
            raise ValueError('Expected one floating trunk')
        self.root_qpos = int(self.model.jnt_qposadr[free[0]])
        self.root_dof = int(self.model.jnt_dofadr[free[0]])
        self.foot_geom_ids = [self._id(mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in ('FL', 'FR', 'RL', 'RR')]
        self.floor_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros_like(self._jacp)
        self.reset()

    def _id(self, kind, name):
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f'Missing model entity: {name}')
        return value

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        i = self.root_qpos
        self.data.qpos[i:i+3] = [0, 0, self.cfg.initial_height]
        yaw = self.cfg.initial_yaw
        # MuJoCo free-joint quaternion is WXYZ. Policy uses R directly.
        self.data.qpos[i+3:i+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        self.data.qpos[self.qpos_ids] = self.default_pos
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def joints(self):
        return self.data.qpos[self.qpos_ids].copy(), self.data.qvel[self.dof_ids].copy()

    def state(self):
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        # COM Jacobian gives physical COM velocity, avoiding link/COM ambiguity.
        mujoco.mj_jacBodyCom(self.model, self.data, self._jacp, self._jacr, self.body_id)
        velocity = rotation.T @ (self._jacp @ self.data.qvel)
        omega = rotation.T @ (self._jacr @ self.data.qvel)
        gravity = rotation.T @ np.array([0., 0., -1.])
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arcsin(np.clip(-rotation[2, 0], -1, 1))
        return dict(position=self.data.xpos[self.body_id].copy(), velocity=velocity, omega=omega,
                    gravity=gravity, roll=float(roll), pitch=float(pitch))

    def observation(self, command, previous_action):
        state = self.state()
        q, dq = self.joints()
        if self.cfg.profile == 'legacy':
            # Gym scales first, then clips the complete observation, unlike Lab.
            return np.clip(np.concatenate((command*np.array([2., 2., .25]), state['omega']*.25,
                          state['gravity'], q-self.default_pos, dq*.05, previous_action)), -100, 100).astype(np.float32)
        command_scale = np.array([1., 1., 1.]) if self.cfg.profile == 'lab' else np.array([2., 2., .25])
        # Lab observation manager clips each raw term BEFORE scaling.
        frame = np.concatenate((np.clip(command, -100, 100) * command_scale,
                                np.clip(state['omega'], -100, 100) * .25,
                                np.clip(state['gravity'], -100, 100),
                                np.clip(q-self.default_pos, -100, 100),
                                np.clip(dq, -100, 100) * .05,
                                np.clip(previous_action, -100, 100)))
        return frame.astype(np.float32)

    def targets(self, action):
        action = np.asarray(action)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('Expected finite 12D action')
        if self.cfg.profile == 'legacy':
            action = np.clip(action, -100, 100)
        scaled = action * self.cfg.controller.action_scale
        scaled = scaled.copy()
        scaled[[i for i, n in enumerate(self.names) if '_hip_' in n]] *= self.cfg.controller.hip_reduction
        # Normal Lab action clipping applies to absolute target, not raw action.
        return np.clip(self.default_pos + scaled, -100, 100)

    def step(self, torque):
        torque = np.asarray(torque)
        if torque.shape != (12,) or not np.isfinite(torque).all():
            raise ValueError('Expected finite 12D joint torque')
        bad_warnings = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                        mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)
        before = [self.data.warning[w].number for w in bad_warnings]
        self.data.ctrl[self.actuator_ids] = torque
        mujoco.mj_step(self.model, self.data)
        # Synchronize derived transforms/contacts with the newly integrated state.
        mujoco.mj_forward(self.model, self.data)
        if any(self.data.warning[w].number > count for w, count in zip(bad_warnings, before)):
            raise FloatingPointError('MuJoCo reported a numerical state/control warning')
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('Non-finite MuJoCo state')

    def contacts(self):
        forces = np.zeros(4)
        trunk = False
        force = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if self.floor_id not in pair:
                continue
            other = pair[1] if pair[0] == self.floor_id else pair[0]
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if other in self.foot_geom_ids:
                forces[self.foot_geom_ids.index(other)] += abs(force[0])
            if self.model.geom_bodyid[other] == self.body_id and abs(force[0]) > 1.:
                trunk = True
        return forces > 1., forces, trunk

    def initial_state_sha256(self):
        return hashlib.sha256(np.concatenate((self.data.qpos, self.data.qvel)).tobytes()).hexdigest()

    def model_metadata(self):
        path = Path(self.cfg.model)
        provenance = path.parent / 'provenance.json'
        return dict(path=str(path.resolve()), xml_sha256=sha256(path),
                    source=json.loads(provenance.read_text()) if provenance.exists() else None,
                    joint_names=list(self.names), qpos_indices=self.qpos_ids.tolist(),
                    qvel_indices=self.dof_ids.tolist(), actuator_indices=self.actuator_ids.tolist(),
                    default_joint_pos=self.default_pos.tolist(), effort_limits=self.effort_limits.tolist(),
                    passive_damping=self.model.dof_damping[self.dof_ids].tolist(),
                    armature=self.model.dof_armature[self.dof_ids].tolist(),
                    frictionloss=self.model.dof_frictionloss[self.dof_ids].tolist(),
                    joint_ranges=self.model.jnt_range[self.joint_ids].tolist(),
                    foot_friction=self.model.geom_friction[self.foot_geom_ids].tolist(),
                    total_mass=float(self.model.body_mass.sum()), timestep=self.model.opt.timestep,
                    torque_motor_verified=True)
