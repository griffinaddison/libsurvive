

(instructions for griffin's desktop only)



hardware checklist before you run:
- does base staion have solid green light? headset and controllers will BOTH be centered in its field of view?
- headset has USB and DP plugged into desktop? has blue lights on front? breakaway connect (middle of headset coord) plugged?
- controllers have solid green light? no light = off; red blinking => critical battery (bad tracking, will die); blue => connecting (i think)
- 



how to run dual arm teleop: valve index knuckles (controllers) -> dual arx L5

1. clone this repo
2. navigate to it `cd ~/libsurvive`
3. enter the arx conda environment `conda activate arx-py310`
4. connect to the arms `sudo systemctl restart arxcan-setup.service`
    1. command runs and terminal prints nothing = success
    2. command runs and terminal prints this `Job for arxcan-setup.service failed because the control process exited with error code.
See "systemctl status arxcan-setup.service" and "journalctl -xeu arxcan-setup.service" for details.`
        5. that means it failed. maybe the arms are off or USB not plugged in.

5. run the teleop script `python bindings/python/examples/teleop-example.py`
6. arms will move to starting position
7. hold each controller such that circular button panel is normal to your line of sight (facing you)
8. hold A button on each controller to engage that arm and have its EE follow your controller pose. Trigger (index finger) closes gripper.
9. CAREFUL! you can easily break the arms, or injure yourself. LIGHTHOUSE NEEDS GOOD VIEW OF CONTROLLERS. 












