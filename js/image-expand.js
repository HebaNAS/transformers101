document.addEventListener('DOMContentLoaded', function() {
  // Select all zoomable images
  const images = document.querySelectorAll('.zoomable-image');
  const overlay = document.getElementById('imageOverlay');
  const expandedImg = document.getElementById('expandedImg');
  const overlayCaption = document.getElementById('overlayCaption');
  const closeButton = document.querySelector('.close-button');
  
  // Add click event to each image
  images.forEach(img => {
    img.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      
      // Get the image source and alt text
      const imgSrc = this.src;
      
      // Get the caption text from the parent li element
      // This extracts the text that comes after the image
      const parentLi = this.closest('li');
      let captionText = '';
      
      if (parentLi) {
        // Get all text nodes directly inside the li element
        const textNodes = Array.from(parentLi.childNodes)
          .filter(node => node.nodeType === Node.TEXT_NODE)
          .map(node => node.textContent.trim())
          .filter(text => text.length > 0);
        
        // If no direct text nodes found, try to get the last text content
        if (textNodes.length === 0) {
          // Get the li's text content and remove any leading/trailing whitespace
          const fullText = parentLi.textContent.trim();
          // Extract just the caption text that appears after the image
          captionText = fullText;
        } else {
          captionText = textNodes.join(' ');
        }
      }
      
      // Set the expanded image source and caption
      expandedImg.src = imgSrc;
      overlayCaption.textContent = captionText;
      
      // Display the overlay
      overlay.style.display = 'block';
      
      // Prevent the click from triggering slide navigation
      return false;
    });
  });
  
  // Close overlay when clicking the close button
  closeButton.addEventListener('click', function() {
    overlay.style.display = 'none';
  });
  
  // Close overlay when clicking outside the image
  overlay.addEventListener('click', function(e) {
    if (e.target === overlay) {
      overlay.style.display = 'none';
    }
  });
  
  // Close overlay when pressing Escape key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      overlay.style.display = 'none';
    }
  });
});

document.addEventListener('DOMContentLoaded', function() {
  // Get all citation icons that are magnifying glasses
  const magnifyingGlassIcons = document.querySelectorAll('.citation-icon');
  
  magnifyingGlassIcons.forEach(icon => {
    if (icon.textContent === '🔍') {
      icon.style.cursor = 'pointer';
      
      icon.addEventListener('click', function() {
        const overlay = document.getElementById('imageOverlay');
        const expandedImg = document.getElementById('expandedImg');
        
        // Determine which visualization to show based on which slide/icon was clicked
        if (this.closest('#section-21')) {
          expandedImg.src = 'images/explanations.png'; // Replace with your actual image path
          document.getElementById('overlayCaption').textContent = 'Semantic Similarity Visualization';
        } else if (this.closest('#section-22')) {
          expandedImg.src = 'images/ip.png'; // Replace with your actual image path
          document.getElementById('overlayCaption').textContent = 'Parameter Analysis Visualization';
        }
        
        overlay.style.display = 'block';
      });
    }
  });
});
