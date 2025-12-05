#include <stdio.h>
#include <string.h>
#include <survive_api.h>
#include <os_generic.h>

static volatile int keepRunning = 1;

#ifdef __linux__

#include <assert.h>
#include <signal.h>
#include <stdlib.h>

void intHandler(int dummy) {
	if (keepRunning == 0)
		exit(-1);
	keepRunning = 0;
}

#endif

static void log_fn(SurviveSimpleContext *actx, SurviveLogLevel logLevel, const char *msg) {
	fprintf(stderr, "(%7.3f) SimpleApi: %s\n", survive_simple_run_time(actx), msg);
}


// Track whether the right controller's A button is engaged
static int right_engaged = 0;
static int have_right_reference_pose = 0;
static SurvivePose right_reference_pose = {0};
static SurvivePose right_reference_pose_inv = {0};

int main(int argc, char **argv) {
#ifdef __linux__
	signal(SIGINT, intHandler);
	signal(SIGTERM, intHandler);
	signal(SIGKILL, intHandler);
#endif

	SurviveSimpleContext *actx = survive_simple_init_with_logger(argc, argv, log_fn);
	if (actx == 0) // implies -help or similiar
		return 0;

	double start_time = OGGetAbsoluteTime();
	survive_simple_start_thread(actx);

	for (const SurviveSimpleObject *it = survive_simple_get_first_object(actx); it != 0;
		 it = survive_simple_get_next_object(actx, it)) {
		printf("Found '%s'\n", survive_simple_object_name(it));
	}

    struct SurviveSimpleEvent event = {0};
	while (keepRunning && survive_simple_wait_for_event(actx, &event) != SurviveSimpleEventType_Shutdown) {
		switch (event.event_type) {
		case SurviveSimpleEventType_PoseUpdateEvent: {
				// only process pose events if right controller's A button is engaged
				if (right_engaged) {
					const struct SurviveSimplePoseUpdatedEvent *pose_event = survive_simple_get_pose_updated_event(&event);
					if (pose_event == 0)
						break;

					if (survive_simple_object_get_subtype(pose_event->object) != SURVIVE_OBJECT_SUBTYPE_KNUCKLES_R)
						break;

					SurvivePose pose = pose_event->pose;
					FLT timecode = pose_event->time;

					if (have_right_reference_pose) {
						// rel_pose = inverse(initial) * current
						SurvivePose rel_pose;
						ApplyPoseToPose(&rel_pose, &right_reference_pose_inv, &pose);
						pose = rel_pose;
					}

					printf("%s %s (%7.3f): \n %.2f %.2f %.2f \n%.2f %.2f %.2f %.2f\n",
						   survive_simple_object_name(pose_event->object), survive_simple_serial_number(pose_event->object),
						   timecode, pose.Pos[0], pose.Pos[1], pose.Pos[2], pose.Rot[0], pose.Rot[1], pose.Rot[2], pose.Rot[3]);
				}
				break;
			}
			case SurviveSimpleEventType_ButtonEvent: {
				const struct SurviveSimpleButtonEvent *button_event = survive_simple_get_button_event(&event);
				if (button_event == 0 || button_event->object == 0)
					break;

				SurviveObjectSubtype subtype = survive_simple_object_get_subtype(button_event->object);

				// only process right knuckle events
				if (subtype != SURVIVE_OBJECT_SUBTYPE_KNUCKLES_R) {
					break;
				}

				if (button_event->button_id == SURVIVE_BUTTON_A) {
					if (button_event->event_type == SURVIVE_INPUT_EVENT_BUTTON_DOWN) {
						right_engaged = 1;
						survive_simple_object_get_latest_pose(button_event->object, &right_reference_pose);
						InvertPose(&right_reference_pose_inv, &right_reference_pose);
						have_right_reference_pose = 1;
						printf("Right A button engaged\n");
					} else if (button_event->event_type == SURVIVE_INPUT_EVENT_BUTTON_UP) {
						right_engaged = 0;
						have_right_reference_pose = 0;
						right_reference_pose = LinmathPose_Identity;
						right_reference_pose_inv = LinmathPose_Identity;
						printf("Right A button released\n");
					}
				}

				break;
			}
		// case SurviveSimpleEventType_ConfigEvent: {
		// 	const struct SurviveSimpleConfigEvent *cfg_event = survive_simple_get_config_event(&event);
		// 	printf("(%f) %s received configuration of length %u type %d-%d\n", cfg_event->time,
		// 		   survive_simple_object_name(cfg_event->object), (unsigned)strlen(cfg_event->cfg),
		// 		   survive_simple_object_get_type(cfg_event->object),
		// 		   survive_simple_object_get_subtype(cfg_event->object));
		// 	break;
		// }
		case SurviveSimpleEventType_DeviceAdded: {
			const struct SurviveSimpleObjectEvent *obj_event = survive_simple_get_object_event(&event);
			printf("(%f) Found '%s'\n", obj_event->time, survive_simple_object_name(obj_event->object));
			break;
		}
		case SurviveSimpleEventType_None:
			break;
		}
	}

	printf("Cleaning up\n");
	survive_simple_close(actx);
	return 0;
}
