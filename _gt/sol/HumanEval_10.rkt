#lang racket
(define (make_palindrome s)
  (define (palindrome? str)
    (string=? str (list->string (reverse (string->list str)))))
  (define n (string-length s))
  (let loop ([i 0])
    (cond [(>= i n) s]
          [(palindrome? (substring s i n))
           (string-append s (list->string (reverse (string->list (substring s 0 i)))))]
          [else (loop (+ i 1))])))
