// This script dynamically sets the 'active' class on the sidebar navigation item
// based on the current page's URL. This ensures the correct menu item is highlighted
// when the page loads and provides a consistent active state across the application.

document.addEventListener('DOMContentLoaded', function() {
    // Select all navigation list items in the sidebar
    const navItems = document.querySelectorAll('.nav-item');

    // Get the filename of the current page from the URL
    // e.g., for 'http://example.com/dashboard.html', currentPath will be 'dashboard.html'
    const currentPath = window.location.pathname.split('/').pop();

    // Iterate over each navigation item
    navItems.forEach(item => {
        // Find the anchor tag (link) within the current navigation item
        const link = item.querySelector('a');

        // Check if a link exists for the item
        if (link) {
            // Get the href attribute of the link
            const href = link.getAttribute('href');

            // Compare the link's href with the current page's filename
            // If they match, add the 'active' class to the navigation item
            if (href === currentPath) {
                item.classList.add('active');
            }

            // Optional: Add a click event listener to each navigation item.
            // This is primarily for visual feedback if a user clicks on the
            // already active link (though it won't reload the page).
            // It ensures that if new content were loaded dynamically without a full page refresh,
            // the active class could still be managed.
            item.addEventListener('click', function() {
                // Remove 'active' class from all other navigation items
                navItems.forEach(nav => nav.classList.remove('active'));
                // Add 'active' class to the clicked item
                this.classList.add('active');
            });
        }
    });
<<<<<<< HEAD
    fetch('/refinement.html', { /* ... options ... */ })
    .then(response => response.json())
    .then(data => {
        // 1. Check the raw data from the server
        console.log("Data received from server:", data);
        console.log("Markdown string:", data.refined_markdown);

        // 2. This is where the error likely is.
        // Check the function that parses the markdown into an array.
        const tableDataArray = parseMarkdownToJSArray(data.refined_markdown);

        // 3. Check the array BEFORE it goes to the table
        console.log("Parsed array before sending to table:", tableDataArray);
        console.log("Number of rows:", tableDataArray.length); // <-- This will probably show 2 instead of 21

        // 4. Initialize the table
        $('#myTable').DataTable({
            data: tableDataArray,
            // ... other options
        });
    });
=======
>>>>>>> f2932ab2d4915c891891df3692f53c87a830cb1f
});