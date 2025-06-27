let img_arr = imgs;
// let img_prefix = 'male_manual_labeling_imgs/'
let data = {}

let img1 = document.getElementById('image1');
let img2 = document.getElementById('image2');
let counter = document.getElementById('counter');

let current_img_idx=0;
interpolate()

function interpolate() {
  let slider = document.getElementById('mySlider');
  let sliderValue = document.getElementById('sliderValue');
  sliderValue.innerHTML = slider.value
  get_image(img_arr[current_img_idx], slider.value)
}

// document.addEventListener("keydown", handleKeyPress);
$.ajax({
  type: "GET",
  url: "/data",
  success: function(response) {
      // console.log(response);
      // $('#dataDisplay').html(JSON.stringify(response));
      data = response
      
      if (img_arr[0] in data) {
        set_signal_values(0)
        update()
      }

      alert('Saved data fetched successfully');
  },
  error: function(xhr, status, error) {
      console.error("Error occurred: " + status + "\nError: " + error);
      alert('Error fetching data');
  }
});

function get_image(imageId, step) {
  $.ajax({
    url: '/get_image',
    type: 'GET',
    data: {
      image_id: imageId,
      step_num: step
    },
    xhrFields: {
        responseType: 'blob'
    },
    success: function(data) {
        const imageUrl = URL.createObjectURL(data);
        img2.src = imageUrl;
    },
    error: function() {
        console.error('There has been a problem with your fetch operation.');
    }
  }); 
}

function update() {
  let counter = document.getElementById('counter');
  counter.innerHTML = current_img_idx.toString() + ' of ' + img_arr.length.toString();
  img1.src = "static/images/" + img_arr[current_img_idx] + "_female.png"
  get_image(img_arr[current_img_idx], 0)

  if (img_arr[current_img_idx] in data) {
    img1.classList.add('done')
    img2.classList.add('done')

  } else {
    img1.classList.remove('done');
    img2.classList.remove('done');

  }

  if (img_arr[current_img_idx] in data) {
    set_signal_values(current_img_idx)
  }
  else {
    set_signal_values_default()
  }
}

function goToIndex() {
  imgIndex = document.getElementById('input').value
  if (isNaN(imgIndex) | parseInt(imgIndex) < 0 | parseInt(imgIndex) >= 1000) {
    alert("Invalid input")
  }
  else {
    update_signals_dict()
    current_img_idx = parseInt(imgIndex);
    update();
  }
  document.getElementById('input').value = ''
  
  
}

function previous() {
  if (current_img_idx > 0) {
    current_img_idx -= 1;
    update();
  }
}

function next() {
  if (current_img_idx < img_arr.length - 1) {
    update_signals_dict()
    current_img_idx += 1;
    update();
  }
}

function set_signal_values(current_img_idx) {
  $(`input[type='radio'][name='A'][value='${data[img_arr[current_img_idx]]['A_realistic']}']`).prop('checked', true);
  $(`input[type='radio'][name='B'][value='${data[img_arr[current_img_idx]]['B_realistic']}']`).prop('checked', true);

  // $(`input[type='radio'][name='shoulders_high_up'][value='${data[img_arr[current_img_idx]]['shoulders_high_up']}']`).prop('checked', true);
  // $(`input[type='radio'][name='text_markers_top_left'][value='${data[img_arr[current_img_idx]]['text_markers_top_left']}']`).prop('checked', true);
  // $(`input[type='radio'][name='text_markers_top_right'][value='${data[img_arr[current_img_idx]]['text_markers_top_right']}']`).prop('checked', true);
  // $(`input[type='radio'][name='l_markers'][value='${data[img_arr[current_img_idx]]['l_markers']}']`).prop('checked', true);
  // $(`input[type='radio'][name='r_markers'][value='${data[img_arr[current_img_idx]]['r_markers']}']`).prop('checked', true);
  // $(`input[type='radio'][name='arrows'][value='${data[img_arr[current_img_idx]]['arrows']}']`).prop('checked', true);
  let textarea = document.getElementById("ab_diff")
  textarea.value = data[img_arr[current_img_idx]]['ab_diff']

  let textarea2 = document.getElementById("ba_diff")
  textarea2.value = data[img_arr[current_img_idx]]['ba_diff']
}

function set_signal_values_default() {
  $(`input[type='radio'][name='A'][value='0']`).prop('checked', true);
  $(`input[type='radio'][name='B'][value='0']`).prop('checked', true);
  // $(`input[type='radio'][name='text_markers_top_left'][value='0']`).prop('checked', true);
  // $(`input[type='radio'][name='text_markers_top_right'][value='0']`).prop('checked', true);
  // $(`input[type='radio'][name='l_markers'][value='0']`).prop('checked', true);
  // $(`input[type='radio'][name='r_markers'][value='0']`).prop('checked', true);
  // $(`input[type='radio'][name='arrows'][value='0']`).prop('checked', true);
  let textarea = document.getElementById("ab_diff")
  textarea.value = ""
  textarea.placeholder = "Enter here..."

  let textarea2 = document.getElementById("ba_diff")
  textarea2.value = ""
  textarea2.placeholder = "Enter here..."


}

function update_signals_dict() {
  var A_realistic = document.querySelector('input[name="A"]:checked').value;
  var B_realistic = document.querySelector('input[name="B"]:checked').value;
  // var text_markers_top_left = document.querySelector('input[name="text_markers_top_left"]:checked').value;
  // var text_markers_top_right = document.querySelector('input[name="text_markers_top_right"]:checked').value;
  // var l_markers = document.querySelector('input[name="l_markers"]:checked').value;
  // var r_markers = document.querySelector('input[name="r_markers"]:checked').value;
  // var arrows = document.querySelector('input[name="arrows"]:checked').value;
  let textarea = document.getElementById("ab_diff")
  let textarea2 = document.getElementById("ba_diff")


  data[img_arr[current_img_idx]] = {'A_realistic': parseInt(A_realistic),
                                    'B_realistic': parseInt(B_realistic),
                                    'ab_diff': textarea.value,
                                    'ba_diff': textarea2.value

                                  };
}

$('#submitButton').click(function(e) {
    update_signals_dict()
    e.preventDefault()
    var $this = $(this); // Cache this for later use
    var originalText = $this.text(); // Save the original button text

    $this.text('Submitting...');
    $.ajax({
      type: "POST",
      url: "/submit",
      contentType: "application/json",
      data: JSON.stringify(data),
      success: function(response) {
          console.log(response);
          alert('Data saved successfully');
      },
      error: function(xhr, status, error) {
          console.error("Error occurred: " + status + "\nError: " + error);
          alert('Error sending data');
      },
      complete: function() {
          // Revert button text back to original after AJAX call is complete
          $this.text(originalText);
      }
    });
})


update();

// $(document).ready(function() {
  
// })