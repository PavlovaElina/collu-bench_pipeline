#lang racket
(define (encode message)
  (define (swap c)
    (cond [(char-upper-case? c) (char-downcase c)]
          [(char-lower-case? c) (char-upcase c)]
          [else c]))
  (define (vowel? c) (member (char-downcase c) '(#\a #\e #\i #\o #\u)))
  (define (shift2 c)
    (cond [(char<=? #\a c #\z) (integer->char (+ 97 (modulo (+ (- (char->integer c) 97) 2) 26)))]
          [(char<=? #\A c #\Z) (integer->char (+ 65 (modulo (+ (- (char->integer c) 65) 2) 26)))]
          [else c]))
  (list->string
   (map (lambda (c) (define sc (swap c)) (if (vowel? sc) (shift2 sc) sc))
        (string->list message))))
