import carla
import cv2
import numpy as np

import os
import random
import time
from queue import Queue


class Ego:
    def __init__(
        self,
        experiment,
        client,
        process_func,
        autopilot,
        camera_resolution,
        save_orig,
        save_processed,
        save_controls,
        save_training_data,
    ):
        self.experiment = experiment
        self.exp_path = os.path.join("jbnav_experiments", self.experiment)
        self.client = client
        self.autopilot = autopilot
        self.process_func = process_func
        self.save_orig = save_orig
        self.images = []
        self.save_processed = save_processed
        self.processed_images = []
        self.save_controls = save_controls
        self.controls = []
        self.save_training_data = save_training_data
        self.training_data = {"images": [], "controls": []}
        self.queue = Queue()

        self.vehicle = None
        self.cam = None
        self.spec_set = False

        world = client.get_world()
        ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "ego")
        ego_color = random.choice(ego_bp.get_attribute("color").recommended_values)
        ego_bp.set_attribute("color", ego_color)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if 0 < number_of_spawn_points:
            random.shuffle(spawn_points)
            ego_transform = spawn_points[0]
            self.vehicle = world.spawn_actor(ego_bp, ego_transform)
        else:
            logging.warning("Could not found any spawn points")

        cam_bp = None
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(camera_resolution[0]))
        cam_bp.set_attribute("image_size_y", str(camera_resolution[1]))
        cam_bp.set_attribute("fov", str(60))
        cam_location = carla.Location(2, 0, 1.5)
        cam_rotation = carla.Rotation(0, 0, 0)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        self.cam = world.spawn_actor(
            cam_bp,
            cam_transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.cam.listen(self.queue.put)

        self.spectator = world.get_spectator()

        if self.autopilot:
            self.vehicle.set_autopilot(True)

    def step(self, n):
        if self.save_controls and n > 1:
            self.controls.append(self._control_to_dict(self.vehicle.get_control()))

        if not self.spec_set:
            self.spectator.set_transform(self.vehicle.get_transform())
            self.spec_set = True

        image = self.queue.get()

        # BGR Format
        np_image = (
            np.array(image.raw_data)
            .astype("uint8")
            .reshape(image.height, image.width, 4)[:, :, 0:-1]
        )

        if self.save_orig:
            self.images.append(np.copy(np_image))

        if self.save_training_data:
            self.training_data["images"].append(np.copy(np_image))
            if n > 1:
                self.training_data["controls"].append(
                    self._control_to_dict(self.vehicle.get_control())
                )

        if self.process_func:
            process_ret = self.process_func(np_image)

            if isinstance(process_ret, tuple):
                output_image, output_control = process_ret[0], process_ret[1]
            elif isinstance(process_ret, carla.VehicleControl):
                output_image, output_control = None, process_ret
            else:
                output_image, output_control = process_ret, None

            if self.save_processed and output_image is not None:
                self.processed_images.append(output_image)

            if not self.autopilot and output_control is not None:
                self.vehicle.apply_control(output_control)

    def cleanup(self):
        if self.save_controls:
            self.controls.append(self._control_to_dict(self.vehicle.get_control()))

        if len(self.images) or len(self.controls) or len(self.processed_images):
            os.makedirs(self.exp_path)

        if self.save_training_data:
            self.training_data["controls"].append(
                self._control_to_dict(self.vehicle.get_control())
            )
            import pickle

            with open(
                os.path.join(self.exp_path, "training_data.pickle"), "wb"
            ) as handle:
                pickle.dump(self.training_data, handle)

        if len(self.images):
            out_video = os.path.join(self.exp_path, "orig_images.mp4")
            image_shape = self.images[0].shape

            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            out = cv2.VideoWriter(
                out_video, fourcc, 1 / 0.05, (image_shape[1], image_shape[0])
            )
            self._save_video_from_images(out, self.images)
            out.release()

        if len(self.processed_images):
            out_video = os.path.join(self.exp_path, "processed_images.mp4")
            image_shape = self.processed_images[0].shape

            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            if len(image_shape) == 2:
                out = cv2.VideoWriter(
                    out_video, fourcc, 1 / 0.05, (image_shape[1], image_shape[0]), 0
                )
            else:
                out = cv2.VideoWriter(
                    out_video,
                    fourcc,
                    1 / 0.05,
                    (image_shape[1], image_shape[0]),
                )
            self._save_video_from_images(out, self.processed_images)
            out.release()

        if len(self.controls) > 0:
            import jsonlines

            with jsonlines.open(
                os.path.join(self.exp_path, "controls.jsonl"),
                "w",
            ) as writer:
                writer.write_all(self.controls)

        if self.cam:
            self.cam.stop()
            self.cam.destroy()
        if self.vehicle:
            self.client.apply_batch([carla.command.DestroyActor(self.vehicle)])


    def _save_video_from_images(self, writer, images):
        for i in range(len(images)):
            img = images[i]
            if img.max() <= 1:
                img *= 255
            writer.write(img)

    def _control_to_dict(self, control):
        return {
            "throttle": control.throttle,
            "steer": control.steer,
            "brake": control.brake,
            "hand_brake": control.hand_brake,
            "reverse": control.reverse,
            "manual_gear_shift": control.manual_gear_shift,
            "gear": control.gear,
        }
