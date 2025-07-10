// self-attention-viz.js - DEBUG VERSION
class SelfAttentionVisualization {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.tokens = [];
        this.connections = [];
        this.animationId = null;
        this.time = 0;
        this.isInitialized = false;
    }

    init() {
        // Skip initialization if running in headless mode (like decktape)
        if (typeof window !== 'undefined' && window.navigator && window.navigator.webdriver) {
            console.log('Skipping Three.js initialization in headless mode');
            return false;
        }

        this.container = document.getElementById(this.containerId);
        if (!this.container) {
            console.error(`Container with id '${this.containerId}' not found`);
            return false;
        }

        // Remove any existing canvas elements
        const existingCanvas = this.container.querySelector('canvas');
        if (existingCanvas) {
            existingCanvas.remove();
        }

        // Get container dimensions
        const rect = this.container.getBoundingClientRect();
        
        if (rect.width === 0 || rect.height === 0) {
            console.error('Container has zero dimensions');
            return false;
        }

        // Check if Three.js is loaded
        if (typeof THREE === 'undefined') {
            console.error('THREE.js not loaded!');
            return false;
        }

        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);

        // Camera setup - medium zoom level
        const aspect = rect.width / rect.height;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(0, 0, 3.25);  // Medium zoom: between 2.5 and 4
        this.camera.lookAt(0, 0, 0);

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(rect.width, rect.height);
        this.renderer.setClearColor(0xffffff, 0);
        
        // Style the canvas
        this.renderer.domElement.style.display = 'block';
        this.renderer.domElement.style.width = '100%';
        this.renderer.domElement.style.height = '100%';
        
        // Add canvas to container
        this.container.appendChild(this.renderer.domElement);

        // Create actual content
        this.createTokens();
        this.createConnections();

        // Force initial render
        this.renderer.render(this.scene, this.camera);

        // Start animation loop
        this.isInitialized = true;
        this.animate();
        
        return true;
    }

    createTokens() {
        const words = ['Your', 'journey', 'starts', 'here'];
        const positions = [
            new THREE.Vector3(-3, 0, 0),
            new THREE.Vector3(-1, 0, 0),
            new THREE.Vector3(1, 0, 0),
            new THREE.Vector3(3, 0, 0)
        ];

        words.forEach((word, index) => {
            // Create sphere for token
            const geometry = new THREE.SphereGeometry(0.4, 16, 16);
            const material = new THREE.MeshBasicMaterial({ 
                color: 0x003f5c,
                transparent: true,
                opacity: 0.8
            });
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.copy(positions[index]);

            // Create text sprite
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 256;
            canvas.height = 128;

            // Clear canvas with white background
            context.fillStyle = '#ffffff';
            context.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw text
            context.fillStyle = '#003f5c';
            context.font = 'bold 32px Arial';
            context.textAlign = 'center';
            context.textBaseline = 'middle';
            context.fillText(word, canvas.width / 2, canvas.height / 2);

            const texture = new THREE.CanvasTexture(canvas);
            texture.needsUpdate = true;
            
            const spriteMaterial = new THREE.SpriteMaterial({ 
                map: texture,
                transparent: true
            });
            const sprite = new THREE.Sprite(spriteMaterial);
            sprite.scale.set(2.5, 1.25, 1);
            sprite.position.copy(positions[index]);
            sprite.position.y += 1.8;  // Increased spacing from 1.2 to 1.8

            // Add to scene
            this.scene.add(sphere);
            this.scene.add(sprite);

            this.tokens.push({
                sphere: sphere,
                sprite: sprite,
                position: positions[index].clone(),
                word: word
            });
        });
    }

    createConnections() {
        // Create connections between all token pairs
        for (let i = 0; i < this.tokens.length; i++) {
            for (let j = 0; j < this.tokens.length; j++) {
                if (i !== j) {
                    const start = this.tokens[i].position;
                    const end = this.tokens[j].position;

                    // Create curved path
                    const distance = Math.abs(i - j);
                    const curveHeight = distance * 0.8;
                    
                    const curve = new THREE.QuadraticBezierCurve3(
                        start,
                        new THREE.Vector3(
                            (start.x + end.x) / 2,
                            curveHeight,
                            0
                        ),
                        end
                    );

                    const points = curve.getPoints(50);
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);

                    // Vary opacity based on distance
                    const baseOpacity = distance === 1 ? 0.6 : (distance === 2 ? 0.4 : 0.3);
                    
                    const material = new THREE.LineBasicMaterial({
                        color: 0xd32f2f,
                        transparent: true,
                        opacity: baseOpacity * 0.7
                    });

                    const line = new THREE.Line(geometry, material);
                    this.scene.add(line);

                    this.connections.push({
                        line: line,
                        material: material,
                        baseOpacity: baseOpacity,
                        index: this.connections.length
                    });
                }
            }
        }
    }

    animate() {
        if (!this.isInitialized) return;

        this.animationId = requestAnimationFrame(() => this.animate());
        this.time += 0.02;  // Gentle animation speed

        // Animate connection opacity (pulsing effect)
        this.connections.forEach((connection) => {
            const pulseOffset = (connection.index * 0.3) + this.time;
            const pulse = (Math.sin(pulseOffset) + 1) * 0.5;
            connection.material.opacity = connection.baseOpacity * (0.4 + pulse * 0.4);
        });

        // Gentle floating animation for tokens
        this.tokens.forEach((token, index) => {
            const offset = index * Math.PI * 0.5;
            const floatY = Math.sin(this.time + offset) * 0.15;
            token.sphere.position.y = token.position.y + floatY;
            token.sprite.position.y = token.position.y + floatY + 1.8;  // Updated to match increased spacing
        });

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        if (this.scene) {
            // Clean up Three.js objects
            this.scene.traverse((object) => {
                if (object.geometry) object.geometry.dispose();
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach(mat => mat.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            });
        }

        if (this.renderer) {
            this.renderer.dispose();
        }

        this.isInitialized = false;
    }

    resize() {
        if (!this.container || !this.camera || !this.renderer) return;

        const rect = this.container.getBoundingClientRect();
        this.camera.aspect = rect.width / rect.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(rect.width, rect.height);
    }
}

// Global instance
let selfAttentionViz = null;

// Initialize visualization
function initSelfAttentionViz() {
    // Destroy existing instance
    if (selfAttentionViz) {
        selfAttentionViz.destroy();
        selfAttentionViz = null;
    }

    // Create new instance
    selfAttentionViz = new SelfAttentionVisualization('self-attention-viz');
    
    // Try to initialize immediately
    const success = selfAttentionViz.init();
    if (!success) {
        selfAttentionViz = null;
    }
    return success;
}

// Check if slide is visible and initialize accordingly
function checkSelfAttentionSlide() {
    const slide = document.getElementById('section-13');
    const container = document.getElementById('self-attention-viz');
    
    if (slide && container) {
        // More comprehensive visibility check
        const isVisible = slide.classList.contains('current') || 
                         slide.classList.contains('show') ||
                         slide.classList.contains('active') ||
                         (getComputedStyle(slide).display !== 'none' && 
                          getComputedStyle(slide).visibility !== 'hidden');
        
        if (isVisible) {
            if (!selfAttentionViz || !selfAttentionViz.isInitialized) {
                initSelfAttentionViz();
            }
        } else {
            if (selfAttentionViz && selfAttentionViz.isInitialized) {
                selfAttentionViz.destroy();
                selfAttentionViz = null;
            }
        }
    }
}

// Setup event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Immediate test
    setTimeout(() => {
        initSelfAttentionViz();
    }, 1000);
    
    // Also set up periodic checking
    setTimeout(checkSelfAttentionSlide, 2000);
    setInterval(checkSelfAttentionSlide, 3000);
    
    // Handle window resize
    window.addEventListener('resize', () => {
        if (selfAttentionViz && selfAttentionViz.isInitialized) {
            selfAttentionViz.resize();
        }
    });
});

// Export for manual initialization
window.initSelfAttentionViz = initSelfAttentionViz;
window.checkSelfAttentionSlide = checkSelfAttentionSlide;

// Auto-initialize after 2 seconds as backup
setTimeout(() => {
    if (!selfAttentionViz || !selfAttentionViz.isInitialized) {
        initSelfAttentionViz();
    }
}, 2000);

// Force initialization when slide becomes visible
function forceInitOnSlideVisible() {
    const targetNode = document.body;
    const config = { attributes: true, childList: true, subtree: true, attributeFilter: ['class'] };
    
    const callback = function(mutationsList, observer) {
        for (let mutation of mutationsList) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                const slide14 = document.getElementById('section-13');
                if (slide14 && (slide14.classList.contains('current') || slide14.classList.contains('show'))) {
                    setTimeout(() => {
                        if (!selfAttentionViz || !selfAttentionViz.isInitialized) {
                            initSelfAttentionViz();
                        }
                    }, 100);
                }
            }
        }
    };
    
    const observer = new MutationObserver(callback);
    observer.observe(targetNode, config);
}

// Set up the mutation observer
document.addEventListener('DOMContentLoaded', () => {
    forceInitOnSlideVisible();
});
