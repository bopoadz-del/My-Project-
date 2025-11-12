// Three.js 3D Visualization for Pendulum

let scene, camera, renderer, pendulumGroup, wireframeMode = false, autoRotate = true;

// Scene dimensions
const CONTAINER_WIDTH = document.querySelector('.threejs-container').clientWidth;
const CONTAINER_HEIGHT = document.querySelector('.threejs-container').clientHeight;

// Initialize Three.js Scene
function initThreeJS() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f1621);
    scene.fog = new THREE.Fog(0x0f1621, 100, 200);

    // Camera
    camera = new THREE.PerspectiveCamera(75, CONTAINER_WIDTH / CONTAINER_HEIGHT, 0.1, 1000);
    camera.position.set(8, 5, 8);
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(CONTAINER_WIDTH, CONTAINER_HEIGHT);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowShadowMap;
    document.querySelector('.threejs-container').appendChild(renderer.domElement);

    // Lighting
    setupLighting();

    // Pendulum Group
    pendulumGroup = new THREE.Group();
    scene.add(pendulumGroup);

    // Add Ground/Base
    addGround();

    // Add Pendulum
    createPendulum();

    // Grid Helper
    const gridHelper = new THREE.GridHelper(20, 20, 0x4ECDC4, 0x2a6f63);
    gridHelper.position.y = -1;
    scene.add(gridHelper);

    // Animation loop
    animate();

    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

// Lighting Setup
function setupLighting() {
    // Directional light (sun)
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(10, 15, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.far = 50;
    scene.add(dirLight);

    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Accent light
    const pointLight = new THREE.PointLight(0x4ECDC4, 0.8);
    pointLight.position.set(-10, 8, -10);
    scene.add(pointLight);
}

// Add Ground
function addGround() {
    const groundGeom = new THREE.PlaneGeometry(30, 30);
    const groundMat = new THREE.MeshStandardMaterial({
        color: 0x1a2635,
        roughness: 0.8,
        metalness: 0.2
    });
    const ground = new THREE.Mesh(groundGeom, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -1;
    ground.receiveShadow = true;
    scene.add(ground);
}

// Create Pendulum Structure
function createPendulum() {
    // Base/Hinge
    const hingeGeom = new THREE.CylinderGeometry(0.3, 0.3, 0.2, 32);
    const hingeMat = new THREE.MeshStandardMaterial({
        color: 0xFFD700,
        metalness: 0.8,
        roughness: 0.2
    });
    const hinge = new THREE.Mesh(hingeGeom, hingeMat);
    hinge.castShadow = true;
    hinge.receiveShadow = true;
    hinge.position.y = 0;
    pendulumGroup.add(hinge);

    // Upper Arm
    const upperArmGeom = new THREE.BoxGeometry(0.4, 3.0, 0.3);
    const armMat = new THREE.MeshStandardMaterial({
        color: 0x4ECDC4,
        metalness: 0.3,
        roughness: 0.4
    });
    const upperArm = new THREE.Mesh(upperArmGeom, armMat);
    upperArm.castShadow = true;
    upperArm.receiveShadow = true;
    upperArm.position.y = 1.5;
    upperArm.name = 'upperArm';
    pendulumGroup.add(upperArm);

    // Lower Arm
    const lowerArmGeom = new THREE.BoxGeometry(0.35, 2.5, 0.3);
    const lowerArm = new THREE.Mesh(lowerArmGeom, armMat);
    lowerArm.castShadow = true;
    lowerArm.receiveShadow = true;
    lowerArm.position.y = -2.0;
    lowerArm.name = 'lowerArm';
    upperArm.add(lowerArm);

    // Tip Weight
    const tipGeom = new THREE.SphereGeometry(0.5, 32, 32);
    const tipMat = new THREE.MeshStandardMaterial({
        color: 0xFF6B6B,
        metalness: 0.6,
        roughness: 0.3
    });
    const tip = new THREE.Mesh(tipGeom, tipMat);
    tip.castShadow = true;
    tip.receiveShadow = true;
    tip.position.y = -2.8;
    tip.name = 'tip';
    lowerArm.add(tip);

    // Vane (Wind-catching surface)
    const vaneGeom = new THREE.BoxGeometry(1.5, 2.5, 0.1);
    const vaneMat = new THREE.MeshStandardMaterial({
        color: 0x44A08D,
        metalness: 0.2,
        roughness: 0.6,
        transparent: true,
        opacity: 0.7
    });
    const vane = new THREE.Mesh(vaneGeom, vaneMat);
    vane.castShadow = true;
    vane.position.z = 1.0;
    vane.position.y = 1.0;
    vane.name = 'vane';
    upperArm.add(vane);

    // Wind Arrow indicator
    createWindArrow();
}

// Create Wind Arrow Indicator
function createWindArrow() {
    const arrowGeom = new THREE.ConeGeometry(0.3, 1.5, 8);
    const arrowMat = new THREE.MeshStandardMaterial({
        color: 0xFF6B6B,
        emissive: 0x990000
    });
    const arrow = new THREE.Mesh(arrowGeom, arrowMat);
    arrow.position.set(-6, 2, 0);
    arrow.rotation.z = Math.PI / 2;
    arrow.name = 'windArrow';
    scene.add(arrow);
}

// Update Pendulum Position
window.updateVisualization = function(frame) {
    if (!pendulumGroup) return;

    const upperArm = pendulumGroup.getObjectByName('upperArm');
    const lowerArm = pendulumGroup.getObjectByName('lowerArm');
    const windArrow = scene.getObjectByName('windArrow');

    if (upperArm) {
        // Rotate upper arm based on theta1
        upperArm.rotation.z = frame.theta1;
    }

    if (lowerArm) {
        // Rotate lower arm based on theta2
        lowerArm.rotation.z = frame.theta2;
    }

    if (windArrow) {
        // Scale arrow based on wind speed
        const windScale = Math.min(frame.wind / 20, 2);
        windArrow.scale.x = 1 + windScale * 0.5;
    }
};

// Animation Loop
function animate() {
    requestAnimationFrame(animate);

    // Auto-rotate camera
    if (autoRotate) {
        camera.position.x = 10 * Math.cos(Date.now() * 0.0003);
        camera.position.z = 10 * Math.sin(Date.now() * 0.0003);
        camera.lookAt(0, 2, 0);
    }

    renderer.render(scene, camera);
}

// Reset Camera
window.resetThreeCamera = function() {
    camera.position.set(8, 5, 8);
    camera.lookAt(0, 0, 0);
};

// Toggle Wireframe
window.toggleWireframeMode = function() {
    wireframeMode = !wireframeMode;
    pendulumGroup.traverse(child => {
        if (child.material) {
            if (Array.isArray(child.material)) {
                child.material.forEach(mat => mat.wireframe = wireframeMode);
            } else {
                child.material.wireframe = wireframeMode;
            }
        }
    });
};

// Toggle Auto Rotate
window.toggleAutoRotate = function() {
    autoRotate = !autoRotate;
};

// Handle Window Resize
function onWindowResize() {
    const width = document.querySelector('.threejs-container').clientWidth;
    const height = document.querySelector('.threejs-container').clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// Initialize on load
window.addEventListener('load', () => {
    setTimeout(initThreeJS, 100); // Slight delay to ensure DOM is ready
});
