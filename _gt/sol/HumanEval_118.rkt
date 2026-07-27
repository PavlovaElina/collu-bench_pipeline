#lang racket
(define (get_closest_vowel word)
  (define (vowel? c) (member (char-downcase c) '(#\a #\e #\i #\o #\u)))
  (define n (string-length word))
  (let loop ([i (- n 2)])
    (cond [(< i 1) ""]
          [(and (vowel? (string-ref word i))
                (not (vowel? (string-ref word (- i 1))))
                (not (vowel? (string-ref word (+ i 1)))))
           (string (string-ref word i))]
          [else (loop (- i 1))])))
