#lang racket
(define (check_if_last_char_is_a_letter txt)
  (define parts (string-split txt " " #:trim? #f))
  (cond [(null? parts) #f]
        [else (define last-w (last parts))
              (and (= (string-length last-w) 1) (char-alphabetic? (string-ref last-w 0)))]))
